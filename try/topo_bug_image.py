"""Topology for PIV computation
===============================

.. autoclass:: TopologyPIV
   :members:
   :private-members:

"""
import logging
import os
import gc
from glob import glob
from fluidimage import get_memory_usage
from fluidimage.topologies.base import TopologyBase
from fluidimage import print_memory_usage

#  from matplotlib.image import imread
from fluiddyn.io.image import imread, imsave

#  from scipy.misc import imread
from fluidimage.topologies.waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
)

import numpy as np

logger = logging.getLogger("fluidimage")


mem_log = {"save": [], "calcul": [], "load": []}


def log(f):
    mem = get_memory_usage()
    mem_log[f].append(mem)


def plot():
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    #  fig = plt.figure()
    #  ax = fig.add_subplot(111)
    del mem_log["load"], mem_log["save"]
    for k, v in mem_log.items():
        plt.plot(v, label=k)

    plt.legend()
    plt.show(block=True)


def save(d):
    log("save")
    k, v = d
    path = "/home/users/vishnu1as/src/fluidimage-cornic/try" + os.path.basename(k)
    imsave(path, v)
    return


def calcul(d):
    log("calcul")
    k, v = d
    v = v ** 2
    return (k, np.transpose(v))


def load(k):
    log("load")
    return (k, imread(k))


class TopologyDebug(TopologyBase):
    """Topology for PIV.

    """

    def __init__(self, n=10):

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

        flist = glob(
            "/home/users/vishnu1as/useful/project/16MILESTONE/Data/"
            "Exp35_2016-06-30_N0.56_L6.0_V0.04_piv3d/PCO_side/level01.*/im*"
        )
        self.wq_load.update({os.path.basename(f): f for f in flist[:n]})

    def compute(self):
        super().compute()
        gc.collect()


if __name__ == "__main__":

    from fluidimage import config_logging

    config_logging("info")
    topology = TopologyDebug(20)
    topology.compute()
    print_memory_usage()

    plot()
    print(mem_log)
