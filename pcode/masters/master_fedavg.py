# -*- coding: utf-8 -*-
import copy
import functools
import time

from torch import distributed as dist
from tqdm.std import trange

import pcode.create_aggregator as create_aggregator
import pcode.create_coordinator as create_coordinator
import pcode.master_utils as master_utils
import pcode.utils.checkpoint as checkpoint
from pcode import create_dataset, create_optimizer
from pcode.masters.master_base import MasterBase
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.logging import save_average_test_stat


class MasterFedAvg(MasterBase):
    def __init__(self, conf):
        super().__init__(conf)
        assert self.conf.fl_aggregate is not None
        # tmp for fake aggregate test loader
        if self.conf.prepare_data == "combine" and self.conf.personal_test:
            self.test_loaders = [self.local_test_loaders[1]]
        # define the aggregators.
        self.aggregator = create_aggregator.Aggregator(
            conf,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.dataset,
            test_loaders=self.test_loaders,
            clientid2arch=self.clientid2arch,
        )
        self.coordinator = create_coordinator.Coordinator(conf, self.metrics)
        conf.logger.log(f"Master initialized the aggregator/coordinator.")
        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )
        # self.participated_ids = set() # selected clients'id from begin

        self.local_models = dict(
            (
                client_id,
                copy.deepcopy(self.master_model)
            )
            for client_id in range(1, 1 + conf.n_clients)
        )
        # GM personal_test method 1
        self.GM_client_coordinators = {} # global model personal perf
        for client_id in self.client_ids:
            self.GM_client_coordinators[client_id] = create_coordinator.Coordinator(conf, self.metrics)
        # GM personal_test method 2
        self.GM_curr_personal_perfs = {}
        self.GM_personal_avg_coordinator = create_coordinator.Coordinator(conf, self.metrics)

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
            
            # detect early stopping.
            self._check_early_stopping()

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

            self.update_local_models(selected_client_ids, flatten_local_models)

            # aggregate the local models and evaluate on the validation dataset.
            self._aggregate_model_and_evaluate(flatten_local_models)
            
            self._update_GM_client_performance()
            
            # evaluate the aggregated model.
            self.conf.logger.log(f"Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()

        self._finishing()
        
            
    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.conf.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average",
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        # uniformly averaged the model before the potential aggregation scheme.
        same_arch = (
            len(self.client_models) == 1
            and self.conf.arch_info["master"] == self.conf.arch_info["worker"][0]
        )

        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        if same_arch:
            fedavg_model = list(fedavg_models.values())[0]
        else:
            fedavg_model = None

        # (smarter) aggregate the model from clients.
        # note that: if conf.fl_aggregate["scheme"] == "federated_average",
        #            then self.aggregator.aggregate_fn = None.
        if self.aggregator.aggregate_fn is not None:
            # evaluate the uniformly averaged model.
            if fedavg_model is not None:
                performance = master_utils.get_avg_perf_on_dataloaders(
                    self.conf,
                    self.coordinator,
                    fedavg_model,
                    self.criterion,
                    self.metrics,
                    self.test_loaders,
                    label=f"fedavg_test_loader",
                )
            else:
                assert "knowledge_transfer" in self.conf.fl_aggregate["scheme"]

                performance = None
                for _arch, _fedavg_model in fedavg_models.items():
                    master_utils.get_avg_perf_on_dataloaders(
                        self.conf,
                        self.coordinator,
                        _fedavg_model,
                        self.criterion,
                        self.metrics,
                        self.test_loaders,
                        label=f"fedavg_test_loader_{_arch}",
                    )

            # aggregate the local models.
            client_models = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                fedavg_model=fedavg_model,
                fedavg_models=fedavg_models,
                flatten_local_models=flatten_local_models,
                performance=performance,
            )
            # here the 'client_models' are updated in-place.
            if same_arch:
                # here the 'master_model' is updated in-place only for 'same_arch is True'.
                self.master_model.load_state_dict(
                    list(client_models.values())[0].state_dict()
                )
            for arch, _client_model in client_models.items():
                self.client_models[arch].load_state_dict(_client_model.state_dict())
        else:
            # update self.master_model in place.
            if same_arch:
                self.master_model.load_state_dict(fedavg_model.state_dict())
            # update self.client_models in place.
            for arch, _fedavg_model in fedavg_models.items():
                self.client_models[arch].load_state_dict(_fedavg_model.state_dict())

        # evaluate the aggregated model on the test data.
        # if same_arch:
        #     master_utils.do_validation(
        #         self.conf,
        #         self.coordinator,
        #         self.master_model,
        #         self.criterion,
        #         self.metrics,
        #         self.test_loaders,
        #         label=f"aggregated_test_loader",
        #     )
        # else:
        #     for arch, _client_model in self.client_models.items():
        #         master_utils.do_validation(
        #             self.conf,
        #             self.coordinator,
        #             _client_model,
        #             self.criterion,
        #             self.metrics,
        #             self.test_loaders,
        #             label=f"aggregated_test_loader_{arch}",
        #         )

    # def _update_client_performance(self, selected_client_ids):
    #     # get client model best performance on personal test distribution
    #     if self.conf.graph.comm_round == 1:
    #         test_client_ids = self.client_ids
    #     else:
    #         test_client_ids = selected_client_ids
    #     for client_id in test_client_ids:
    #         self.curr_personal_perfs[client_id] = master_utils.do_validation_personal(
    #             self.conf,
    #             self.client_coordinators[client_id],
    #             self.local_models[client_id],
    #             self.criterion,
    #             self.metrics,
    #             [self.local_test_loaders[client_id]],
    #             label=f"local_test_loader_client_{client_id}",
    #         )
        
    #     self._compute_best_mean_client_performance()

    def _update_GM_client_performance(self):
        for client_id in self.client_ids:
            self.GM_curr_personal_perfs[client_id] = master_utils.do_validation_personal(
                self.conf,
                self.GM_client_coordinators[client_id],
                self.client_models[self.clientid2arch[client_id]],
                self.criterion,
                self.metrics,
                [self.local_test_loaders[client_id]],
                label=f"GM_local_test_loader_client_{client_id}",
            )
        
        # personal_test method 2
        curr_perf = []
        for curr_personal_perf in self.GM_curr_personal_perfs.values():
            curr_perf.append(curr_personal_perf)
        curr_perf = functools.reduce(lambda a, b: a + b, curr_perf) / len(curr_perf)
        save_average_test_stat(self.conf, curr_perf.dictionary, type="GM_average_test")
        self.GM_personal_avg_coordinator.update_perf(curr_perf)

        for name, best_tracker in self.GM_personal_avg_coordinator.best_trackers.items():
            self.conf.logger.log(
                "GM Personal_test method 2 \
                -- GM Personal best avg results of {}: \t {} best comm round {:.3f}, current comm_roud {:.3f}".format(
                    name,
                    best_tracker.best_perf,
                    best_tracker.get_best_perf_loc,
                    self.conf.graph.comm_round,
                )
            )       
    
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
