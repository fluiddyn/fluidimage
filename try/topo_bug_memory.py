"""Topology for PIV computation
===============================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import os
import logging

from fluidimage import ParamContainer

from fluidimage.topologies.base import TopologyBase

from fluidimage.topologies.waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
)

import numpy as np

logger = logging.getLogger("fluidimage")


def save(o):
    return


def calcul(o):
    return o ** 2


def load(k):
    return np.ones(10000000)


class TopologyDebug(TopologyBase):
    """Topology for PIV.

    """

    def __init__(self, n=10000):

        self.results = {}

        self.wq_save = WaitingQueueThreading(
            "save", save, self.results, topology=self
        )
        self.wq_cpu = WaitingQueueMultiprocessing(
            "cpu", calcul, self.wq_save, topology=self
        )
        self.wq_load = WaitingQueueThreading(
            "load", load, self.wq_cpu, topology=self
        )
        # self.wq_load = WaitingQueueLoadImage(
        #     destination=self.wq_cpu,
        #     path_dir=path_dir, topology=self)

        super().__init__([self.wq_load, self.wq_cpu, self.wq_save])

        self.wq_load.update({i: i for i in range(100)})


if __name__ == "__main__":

    from fluidimage import config_logging

    config_logging("info")

    topology = TopologyDebug()

    topology.compute()
