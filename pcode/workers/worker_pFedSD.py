# -*- coding: utf-8 -*-
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules import loss
from tqdm import tqdm

import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.local_training.random_reinit as random_reinit
import pcode.models as models
from pcode import master_utils
from pcode.utils.logging import display_training_stat
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.workers.worker_base import WorkerBase


class WorkerpFedSD(WorkerBase):
    def __init__(self, conf):
        super().__init__(conf)

        self.M = 1

    def run(self):
        while True:
            self._listen_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_early_stopping():
                return

            self._recv_model_from_master()

            if self.is_active_before == 0:
                self._train()
            else:
                self._train_AKT()
                
            self._send_model_to_master()

            # check if we need to terminate the training or not.
            if self._terminate_by_complete_training():
                return
    
    def _listen_to_master(self):
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((4, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)
        self.conf.graph.client_id, self.conf.graph.comm_round, self.n_local_epochs, self.is_active_before= (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )

        # once we receive the signal, we init for the local training.
        self.arch, self.model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.metrics = create_metrics.Metrics(self.model, task="classification")
        dist.barrier()
    

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        dist.recv(self.model_tb.buffer, src=0)
        self.model_tb.unpack(self.model_state_dict.values())
        self.model.load_state_dict(self.model_state_dict)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received the global/personal model ({self.arch}) from Master."
        )
        
        if self.is_active_before == 1:
            # init the placeholders to recv the other local models from master.
            flatten_local_models = []
            for i in range(self.M):
                client_tb = TensorBuffer(
                    list(copy.deepcopy(self.model.state_dict()).values())
                )
                client_tb.buffer = torch.zeros_like(client_tb.buffer)
                flatten_local_models.append(client_tb)
            # receive local models from master.
            for i in range(self.M):
                dist.recv(tensor=flatten_local_models[i].buffer, src=0)
            
            self.last_local_model = copy.deepcopy(self.model)
            _last_model_state_dict = self.last_local_model.state_dict()
            flatten_local_models[0].unpack(_last_model_state_dict.values())
            self.last_local_model.load_state_dict(_last_model_state_dict)

            self.conf.logger.log(
                f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) received ({self.M}) past local models from Master."
            )
      
        dist.barrier()  

    def _train_AKT(self):
        self.model.train()

        # init the model and dataloader.
        self.prepare_local_train_loader()
        if self.conf.graph.on_cuda:
            self.model = self.model.cuda()
            self.last_local_model = self.last_local_model.cuda()

        self.optimizer = create_optimizer.define_optimizer(
            self.conf, model=self.model, optimizer_name=self.conf.optimizer
        )
        self.scheduler = create_scheduler.Scheduler(self.conf, optimizer=self.optimizer)
        self.tracker = RuntimeTracker(metrics_to_track=self.metrics.metric_names)
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) enters the local training phase (current communication rounds={self.conf.graph.comm_round})."
        )

        # entering local updates and will finish only after reaching the expected local_n_epochs.
        while True:
            for _input, _target in self.train_loader:
                # load data
                with self.timer("load_data", epoch=self.scheduler.epoch_):
                    data_batch = create_dataset.load_data_batch(
                        self.conf, _input, _target, is_training=True
                    )

                # inference and get current performance.
                with self.timer("forward_pass", epoch=self.scheduler.epoch_):
                    self.optimizer.zero_grad()
                    loss, _ = self._local_training_with_last_local_model(data_batch)

                with self.timer("backward_pass", epoch=self.scheduler.epoch_):
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                # display the logging info.
                # display_training_stat(self.conf, self.scheduler, self.tracker)

                # display tracking time.
                if (
                    self.conf.display_tracked_time
                    and self.scheduler.local_index % self.conf.summary_freq == 0
                ):
                    self.conf.logger.log(self.timer.summary())

                # check divergence.
                if self.tracker.stat["loss"].avg > 1e3 or np.isnan(
                    self.tracker.stat["loss"].avg
                ):
                    self.conf.logger.log(
                        f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) diverges!!!!!Early stop it."
                    )
                    self._terminate_comm_round()
                    return

                # check stopping condition.
                if self._is_finished_one_comm_round():
                    display_training_stat(self.conf, self.scheduler, self.tracker)
                    self._terminate_comm_round()
                    return

            display_training_stat(self.conf, self.scheduler, self.tracker)

            # refresh the logging cache at the end of each epoch.
            self.tracker.reset()
            if self.conf.logger.meet_cache_limit():
                self.conf.logger.save_json()

    def _local_training_with_last_local_model(self, data_batch): 
        loss, output = self._inference(data_batch)
        # feature_stu = self.model.activations[-1]
        performance = self.metrics.evaluate(loss, output, data_batch["target"])

        last_local_logit = self.last_local_model(data_batch["input"])
        # feature_teacher = self.last_local_model.activations[-1]
        loss2 = self.conf.lamda * self._divergence(
            student_logits = output,
            teacher_logits = last_local_logit,
            KL_temperature=self.conf.KL_T,
        )

        loss_all = loss + loss2

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance + [loss2.item()], n_samples=data_batch["input"].size(0)
            )
        return loss_all, output    
   
    def _terminate_comm_round(self):
        # if self.conf.graph.comm_round > 1:
        #     if self.check_overfit:
        #         self._check_overfit()
        #     else:
        #         self.model.load_state_dict(self.received_models[-1].state_dict())
        self.model = self.model.cpu()
        if hasattr(self, 'init_model'):
            del self.init_model
        self.scheduler.clean()
        self.conf.logger.save_json()
        self.conf.logger.log(
            f"Worker-{self.conf.graph.worker_id} (client-{self.conf.graph.client_id}) finished one round of federated learning: (comm_round={self.conf.graph.comm_round})."
        )

def sigmoid_rampup(current, rampup_length = 3):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length = 15):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def attention(x):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    """
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()
