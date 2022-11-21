# -*- coding: utf-8 -*-

from pcode.workers.worker_base import WorkerBase


class WorkerFedAvg(WorkerBase):
    def __init__(self, conf):
        super().__init__(conf)
