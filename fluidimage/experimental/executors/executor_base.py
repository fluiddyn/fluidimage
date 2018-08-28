"""Base class for executors
===========================

.. autoclass:: ExecutorBase
   :members:
   :private-members:

"""

import os
import re
from multiprocessing import cpu_count
from fluidimage.config import get_config
from warnings import warn


class ExecutorBase:
    """Base class for executors.

    It defines best numbers of workers for the local computer.

    Parameters
    ----------

    topology : fluidimage.topology

      A Topology from fluidimage.topology.

    """

    def __init__(self, topology):
        self.topology = topology
        self.queues = []
        # assigne dict to queue
        for q in self.topology.queues:
            new_queue = {}
            self.queues.append(new_queue)
            q.queue = self.queues[-1]

        config = get_config()

        # dt = 0.25  # s
        # dt_small = 0.02
        # dt_update = 0.1

        nb_cores = cpu_count()

        if config is not None:
            try:
                allow_hyperthreading = eval(
                    config["topology"]["allow_hyperthreading"]
                )
            except KeyError:
                allow_hyperthreading = True

        try:  # should work on UNIX
            # found in http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python # noqa
            with open("/proc/self/status") as f:
                status = f.read()
            m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", status)
            if m:
                nb_cpus_allowed = bin(int(m.group(1).replace(",", ""), 16)).count(
                    "1"
                )

            if nb_cpus_allowed > 0:
                nb_cores = nb_cpus_allowed

            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()

            nb_proc_tot = 0
            siblings = None
            for line in cpuinfo.split("\n"):
                if line.startswith("processor"):
                    nb_proc_tot += 1
                if line.startswith("siblings") and siblings is None:
                    siblings = int(line.split()[-1])

            if nb_proc_tot == siblings * 2:
                if allow_hyperthreading is False:
                    print("We do not use hyperthreading.")
                    nb_cores //= 2

        except IOError:
            pass

        nb_max_workers = None
        if config is not None:
            try:
                nb_max_workers = eval(config["topology"]["nb_max_workers"])
            except KeyError:
                pass

        # default nb_max_workers
        # Difficult: trade off between overloading and limitation due to input
        # output.  The user can do much better for a specific case.
        if nb_max_workers is None:
            if nb_cores < 16:
                nb_max_workers = nb_cores + 2
            else:
                nb_max_workers = nb_cores

        self.nb_max_workers = nb_max_workers
