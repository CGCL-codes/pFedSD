# -*- coding: utf-8 -*-
import copy
import functools
import os
import time

import numpy as np
import torch
import torch.distributed as dist

import pcode.create_coordinator as create_coordinator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
import pcode.utils.cross_entropy as cross_entropy
from pcode import master_utils
from pcode.utils.logging import save_average_test_stat
from pcode.utils.tensor_buffer import TensorBuffer


class MasterBase(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))

        # create model as well as their corresponding state_dicts.
        master_arch, self.master_model = create_model.define_model(
            conf, to_consistent_model=False
        )
        self.used_client_archs = set(
            [
                create_model.determine_arch(conf, client_id, use_complex_arch=True)
                for client_id in range(1, 1 + conf.n_clients)
            ]
        )
        self.conf.used_client_archs = self.used_client_archs

        conf.logger.log(f"The client will use archs={self.used_client_archs}.")
        conf.logger.log("Master created model templates for client models.")
        self.client_models = dict(
            (arch, copy.deepcopy(self.master_model))
            if arch == master_arch
            else create_model.define_model(conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    conf, client_id=client_id, use_complex_arch=True
                ),
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        self.conf.clientid2arch = self.clientid2arch
        conf.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

        dist.barrier()
        self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        
        if self.conf.prepare_data == "combine" and self.conf.personal_test:
            self.local_datasets = {}
            self.data_partitioner = create_dataset.define_combine_dataset(
                self.conf,
                dataset=self.dataset,
            )
            conf.logger.log(f"Master initialized the local combined data with workers.")

            for client_id in self.client_ids:
                self.local_datasets[client_id] = create_dataset.define_local_dataset(
                    conf, client_id, self.data_partitioner
                )
            conf.logger.log(f"Master initialized the local dataset split with workers.")
            
            # define local data_loader
            self.local_test_loaders = {}

            for client_id in self.client_ids:
                self.local_test_loaders[client_id] = create_dataset.define_local_data_loader(
                    conf,
                    client_id,
                    data_type = "test",
                    data=self.local_datasets[client_id]["test"],
                )
            conf.logger.log(f"Master initialized local test data.")                
        else:
            _, self.data_partitioner = create_dataset.define_data_loader(
                self.conf,
                dataset=self.dataset["train"],
                localdata_id=0,  # random id here.
                is_train=True,
                data_partitioner=None,
            )
            conf.logger.log(f"Master initialized the local training data with workers.")

            # create val loader.
            # right now we just ignore the case of partitioned_by_user.
            if self.dataset["val"] is not None:
                assert not conf.partitioned_by_user
                self.val_loader, _ = create_dataset.define_data_loader(
                    conf, self.dataset["val"], is_train=False
                )
                conf.logger.log(f"Master initialized val data.")
            else:
                self.val_loader = None

            # create test loaders.
            # localdata_id start from 0 to the # of clients - 1. client_id starts from 1 to the # of clients.
            if conf.partitioned_by_user:
                self.test_loaders = []
                for localdata_id in self.client_ids:
                    test_loader, _ = create_dataset.define_data_loader(
                        conf,
                        self.dataset["test"],
                        localdata_id=localdata_id - 1,
                        is_train=False,
                        shuffle=False,
                    )
                    self.test_loaders.append(copy.deepcopy(test_loader))
            else:
                test_loader, _ = create_dataset.define_data_loader(
                    conf, self.dataset["test"], is_train=False
                )
                self.test_loaders = [test_loader]

            if self.conf.personal_test:
                dist.barrier()
                _, self.personal_test_data_partitioner = create_dataset.define_data_loader(
                    conf,
                    self.dataset["test"],
                    localdata_id=0,
                    is_train=False,
                    data_partitioner=None,
                    personal_partition=True,
                    known_distribution=self.data_partitioner.class_distribution,
                )

                self.local_test_loaders = {}
                for client_id in self.client_ids:
                    self.local_test_loaders[client_id], _ = create_dataset.define_data_loader(
                        conf,
                        dataset=self.dataset["test"],
                        localdata_id=client_id - 1,
                        is_train=False,
                        data_partitioner=self.personal_test_data_partitioner,
                        personal_partition=True,
                        known_distribution=self.data_partitioner.class_distribution,
                    )
                conf.logger.log(f"Master initialized local test data.")

        # define the criterion and metrics.
        self.criterion = cross_entropy.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        conf.logger.log(f"Master initialized model/dataset/criterion/metrics.")

        if self.conf.personal_test:
            self.client_coordinators = {}
            for client_id in self.client_ids:
                self.client_coordinators[client_id] = create_coordinator.Coordinator(conf, self.metrics)

            self.curr_personal_perfs = {}
            self.personal_avg_coordinator = create_coordinator.Coordinator(conf, self.metrics)
            conf.logger.log(f"Master initialized personal avg coodinator for every client.\n")

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _activate_selected_clients(
        self, selected_client_ids, comm_round, list_of_local_n_epochs
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((3, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = torch.Tensor(list_of_local_n_epochs)
        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.conf.logger.log(f"Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]
            client_model_state_dict = self.client_models[arch].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(
                f"\tMaster send the global model={arch} to process_id={worker_rank}."
            )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )
            reqs.append(req)
            
        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"Master received all local models.")
        return flatten_local_models
    
    def _compute_best_mean_client_performance(self):
        # personal_test method 2
        curr_perf = []
        for curr_personal_perf in self.curr_personal_perfs.values():
            curr_perf.append(curr_personal_perf)
        curr_perf = functools.reduce(lambda a, b: a + b, curr_perf) / len(curr_perf)
        save_average_test_stat(self.conf, curr_perf.dictionary)
        self.personal_avg_coordinator.update_perf(curr_perf)

        for name, best_tracker in self.personal_avg_coordinator.best_trackers.items():
            self.conf.logger.log(
                "Personal_test method 2 \
                -- Personal best avg results of {}: \t {} best comm round {:.3f}, current comm_roud {:.3f}".format(
                    name,
                    best_tracker.best_perf,
                    best_tracker.get_best_perf_loc,
                    self.conf.graph.comm_round,
                )
            )       

    def _finishing(self, _comm_round=None):
        self.conf.logger.save_json()
        self.conf.logger.log(f"Master finished the federated learning.")
        self.conf.is_finished = True
        self.conf.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")
