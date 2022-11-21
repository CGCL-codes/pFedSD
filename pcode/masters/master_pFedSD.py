# -*- coding: utf-8 -*-
import copy
import functools
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode import create_dataset, create_optimizer
from pcode.masters.master_base import MasterBase
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.tensor_buffer import TensorBuffer


class MasterpFedSD(MasterBase):
    def __init__(self, conf):
        super().__init__(conf)
        assert self.conf.fl_aggregate is not None

        self.local_models = dict(
            (
                client_id,
                copy.deepcopy(self.master_model)
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.personalized_global_models = dict(
            (
                client_id,
                copy.deepcopy(self.master_model)
            )
            for client_id in range(1, 1 + conf.n_clients)
        )

        conf.logger.log(f"Master initialize the local_models")

        self.M = 1
        self.activated_ids = set() # selected clients'id from begin
        # self.is_cluster = False
        self.is_part_update = False

        if self.is_part_update:
            if 'cifar' in conf.data:
                self.head = [self.master_model.weight_keys[i] for i in [2]]
            self.head = list(itertools.chain.from_iterable(self.head))

        # cluster
        # if self.is_cluster:
        #     self.K = 1 # 0...k-1
        #     self.quasi_global_models = dict(
        #         (
        #             client_id,
        #             copy.deepcopy(self.master_model)
        #         )
        #         for client_id in range(self.K)
        #     )

        # save arguments to disk.
        conf.is_finished = False
        checkpoint.save_arguments(conf)

    def run(self):
        flatten_local_models = None
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round
            self.conf.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # get random n_local_epochs.
            list_of_local_n_epochs = master_utils.get_n_local_epoch(
                conf=self.conf, n_participated=self.conf.n_participated
            )
            self.list_of_local_n_epochs = list_of_local_n_epochs

            # random select clients from a pool.
            selected_client_ids = self._random_select_clients()
            if self.is_part_update:
                self._update_personalized_global_models(selected_client_ids)   # partitial update
            
            # detect early stopping.
            # self._check_early_stopping()

            # init the activation tensor and broadcast to all clients (either start or stop).
            self._activate_selected_clients(
                selected_client_ids, self.conf.graph.comm_round, list_of_local_n_epochs
            )

            # will decide to send the model or stop the training.
            if not self.conf.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(selected_client_ids)
            else:
                dist.barrier()
                self.conf.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return
            # wait to receive the local models.
            flatten_local_models = self._receive_models_from_selected_clients(
                selected_client_ids
            )
            self.activated_ids.update(selected_client_ids)

            self.update_local_models(selected_client_ids, flatten_local_models)

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model(selected_client_ids) 
            
            if self.conf.personal_test:
                self._update_client_performance(selected_client_ids)
            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")
            
        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()

        self._finishing()
        
    def _activate_selected_clients(
        self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((4, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)

        is_activate_before = [1 if id in self.activated_ids else 0 for id in selected_client_ids]      
        activation_msg[3, :] = torch.Tensor(is_activate_before)

        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            if not self.is_part_update:
                global_model_state_dict = self.master_model.state_dict()
            else:
                global_model_state_dict = self.personalized_global_models[selected_client_id].state_dict()
                
            flatten_model = TensorBuffer(list(global_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(
                f"\tMaster send the global model to process_id={worker_rank}."
            )

            if selected_client_id in self.activated_ids:
                # send local model
                local_model_state_dict = self.local_models[selected_client_id].state_dict()
                flatten_local_model = TensorBuffer(list(local_model_state_dict.values()))
                dist.send(tensor=flatten_local_model.buffer, dst=worker_rank)
                self.conf.logger.log(
                    f"\tMaster send local models to process_id={worker_rank}."
                )
        dist.barrier()

    def _aggregate_model(self, selected_client_ids):        
        self.master_model.load_state_dict(self._average_model(selected_client_ids).state_dict())

    def _update_personalized_global_models(self, selected_client_ids):
        if self.conf.graph.comm_round == 1:
            return
        w_master = self.master_model.state_dict()
        for id in selected_client_ids:
            w_local = self.local_models[id].state_dict()
            w_personalized_global = copy.deepcopy(w_master)
            for key in w_local.keys():
                if key in self.head:
                    w_personalized_global[key] = w_local[key]
            self.personalized_global_models[id].load_state_dict(w_personalized_global)

    def _average_model(self, client_ids, weights=None):
        _model_avg = copy.deepcopy(self.master_model)
        for param in _model_avg.parameters():
            param.data = torch.zeros_like(param.data)

        for id in client_ids:
            for avg_param, client_param in zip(_model_avg.parameters(), self.local_models[id].parameters()):
                avg_param.data += client_param.data.clone() / len(client_ids)
        return _model_avg

    def _update_client_performance(self, selected_client_ids):
        # get client model best performance on personal test distribution
        if self.conf.graph.comm_round == 1:
            test_client_ids = self.client_ids
        else:
            test_client_ids = selected_client_ids
        for client_id in test_client_ids:
            self.curr_personal_perfs[client_id] = master_utils.do_validation_personal(
                self.conf,
                self.client_coordinators[client_id],
                self.local_models[client_id],
                self.criterion,
                self.metrics,
                [self.local_test_loaders[client_id]],
                label=f"local_test_loader_client_{client_id}",
            )
        
        self._compute_best_mean_client_performance() 
    
    def update_local_models(self, selected_client_ids, flatten_local_models):
        _, local_models = master_utils.recover_models(
            self.conf, self.client_models, flatten_local_models
        )
        for selected_client_id in selected_client_ids:
           self.local_models[selected_client_id].load_state_dict(local_models[selected_client_id].state_dict())
    
    def _check_early_stopping(self):
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                self.coordinator.key_metric.cur_perf is not None
                and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.conf.logger.log("Master early stopping: meet target perf.")
                self.conf.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.conf.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True
        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.conf.graph.comm_round - 1
            self.conf.graph.comm_round = -1
            self._finishing(_comm_round)


def get_model_diff(model1, model2):
    params_dif = []
    for param_1, param_2 in zip(model1.parameters(), model2.parameters()):
        params_dif.append((param_1 - param_2).view(-1))
    params_dif = torch.cat(params_dif)
    return torch.norm(params_dif).item()

