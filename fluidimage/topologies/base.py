"""Topology base (:mod:`fluidimage.topologies.base`)
====================================================

.. autoclass:: TopologyBase
   :members:
   :private-members:

"""
from __future__ import print_function

from time import sleep, time
from multiprocessing import cpu_count, Process
import signal
import re
import sys
import os
import gc

# from copy import copy
import threading
from warnings import warn

try:
    import queue
except ImportError:
    # python 2
    import Queue as queue

from fluiddyn import time_as_str
from fluiddyn.io.tee import MultiFile

from ..util.util import cstring, logger, log_memory_usage
from ..config import get_config
from .waiting_queues.base import WaitingQueueThreading
from .. import config_logging

_stdout_at_import = sys.stdout
_stderr_at_import = sys.stderr

config = get_config()

dt = 0.25  # s
dt_small = 0.02
dt_update = 0.1

if "OMP_NUM_THREADS" not in os.environ:
    warn("OMP_NUM_THREADS not set")
else:
    OMP_NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
    if OMP_NUM_THREADS > 1:
        warn("OMP_NUM_THREADS is greater than 1!")

nb_cores = cpu_count()

if config is not None:
    try:
        allow_hyperthreading = eval(config["topology"]["allow_hyperthreading"])
    except KeyError:
        allow_hyperthreading = True

try:  # should work on UNIX
    # found in http://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python # noqa
    with open("/proc/self/status") as f:
        status = f.read()
    m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", status)
    if m:
        nb_cpus_allowed = bin(int(m.group(1).replace(",", ""), 16)).count("1")

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
# Difficult: trade off between overloading and limitation due to input output.
# The user can do much better for a specific case.
if nb_max_workers is None:
    if nb_cores < 16:
        nb_max_workers = nb_cores + 2
    else:
        nb_max_workers = nb_cores

_nb_max_workers = nb_max_workers


class TopologyBase(object):
    """Base class for topologies of processing.

    Parameters
    ----------

    queues : list

    path_output : None

    logging_level : 'info'

    nb_max_workers : None

    """

    def __init__(
        self, queues, path_output=None, logging_level="info", nb_max_workers=None
    ):

        if path_output is not None:
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            self.path_output = path_output
            log = os.path.join(
                path_output,
                "log_" + time_as_str() + "_" + str(os.getpid()) + ".txt",
            )
            self._log_file = open(log, "w")

            stdout = sys.stdout
            if isinstance(stdout, MultiFile):
                stdout = _stdout_at_import

            stderr = sys.stderr
            if isinstance(stderr, MultiFile):
                stderr = _stderr_at_import

            sys.stdout = MultiFile([stdout, self._log_file])
            sys.stderr = MultiFile([stderr, self._log_file])

        if logging_level is not None:
            for handler in logger.handlers:
                logger.removeHandler(handler)

            config_logging(logging_level, file=sys.stdout)

        if nb_max_workers is None:
            nb_max_workers = _nb_max_workers

        self.nb_max_workers_io = max(int(nb_max_workers * 0.8), 2)
        self.nb_max_launch = max(int(self.nb_max_workers_io), 1)

        if nb_max_workers < 1:
            raise ValueError("nb_max_workers < 1")

        print("nb_cpus_allowed = {}".format(nb_cores))
        print("nb_max_workers = ", nb_max_workers)
        print("nb_max_workers_io = ", self.nb_max_workers_io)

        self.queues = queues
        self.nb_max_workers = nb_max_workers
        self.nb_cores = nb_cores
        self.nb_items_lim = max(2 * nb_max_workers, 2)

        self._has_to_stop = False

        if sys.platform != "win32":

            def handler_signals(signal_number, stack):
                print(
                    "signal {} received: set _has_to_stop to True".format(
                        signal_number
                    )
                )
                self._has_to_stop = True

            signal.signal(12, handler_signals)

    def compute(self, sequential=None, has_to_exit=True):
        """Compute (run all works to be done).

        Parameters
        ----------

        sequential : None

          If bool(sequential) is True, the computations are run in sequential
          (useful for debugging).

        has_to_exit : True

          If bool(has_to_exit) is True and if the computation has to stop
          because of a signal 12 (cluster), a signal 99 is sent at exit.

        """
        if hasattr(self, "path_output"):
            logger.info("path results:\n" + self.path_output)
            if hasattr(self, "params"):
                tmp_path_params = os.path.join(
                    self.path_output,
                    "params_" + time_as_str() + "_" + str(os.getpid()),
                )

                if not os.path.exists(tmp_path_params + ".xml"):
                    path_params = tmp_path_params + ".xml"
                else:
                    i = 1
                    while os.path.exists(tmp_path_params + "_" + str(i) + ".xml"):
                        i += 1
                    path_params = tmp_path_params + "_" + str(i) + ".xml"
                self.params._save_as_xml(path_params)

        self.t_start = time()

        log_memory_usage(time_as_str(2) + ": start compute. mem usage")

        self.nb_workers_cpu = 0
        self.nb_workers_io = 0
        workers = []

        class CheckWorksThread(threading.Thread):
            cls_to_be_updated = threading.Thread

            def __init__(self):
                self.has_to_stop = False
                super(CheckWorksThread, self).__init__()
                self.exitcode = None
                self.daemon = True

            def in_time_loop(self):
                t_tmp = time()
                for worker in workers:
                    if (
                        isinstance(worker, self.cls_to_be_updated)
                        and worker.fill_destination()
                    ):
                        workers.remove(worker)
                t_tmp = time() - t_tmp
                if t_tmp > 0.2:
                    logger.info(
                        "update list of workers with fill_destination "
                        "done in {:.3f} s".format(t_tmp)
                    )
                sleep(dt_update)

            def run(self):
                try:
                    while not self.has_to_stop:
                        self.in_time_loop()
                except Exception as e:
                    print("Exception in UpdateThread")
                    self.exitcode = 1
                    self.exception = e

        class CheckWorksProcess(CheckWorksThread):
            cls_to_be_updated = Process

            def in_time_loop(self):
                # weird bug subprocessing py3
                for worker in workers:
                    if not worker.really_started:
                        # print('check if worker has really started.' +
                        #       worker.key)
                        try:
                            worker.really_started = (
                                worker.comm_started.get_nowait()
                            )
                        except queue.Empty:
                            pass
                        if (
                            not worker.really_started
                            and time() - worker.t_start > 10
                        ):
                            # bug! The worker does not work. We kill it! :-)
                            logger.error(
                                cstring(
                                    "Mysterious bug multiprocessing: "
                                    "a launched worker has not started. "
                                    "We kill it! ({}, key: {}).".format(
                                        worker.work_name, worker.key
                                    ),
                                    color="FAIL",
                                )
                            )
                            # the case of this worker has been
                            worker.really_started = True
                            worker.terminate()

                super(CheckWorksProcess, self).in_time_loop()

        self.thread_check_works_t = CheckWorksThread()
        self.thread_check_works_t.start()

        self.thread_check_works_p = CheckWorksProcess()
        self.thread_check_works_p.start()

        while not self._has_to_stop and (
            any([not q.is_empty() for q in self.queues]) or len(workers) > 0
        ):
            # debug
            # if logger.level == 10 and \
            #    all([q.is_empty() for q in self.queues]) and len(workers) == 1:
            #     for worker in workers:
            #         try:
            #             is_alive = worker.is_alive()
            #         except AttributeError:
            #             is_alive = None

            #         logger.debug(
            #             str((worker, worker.key, worker.exitcode, is_alive)))

            #         if time() - worker.t_start > 60:
            #             from fluiddyn import ipydebug
            #             ipydebug()

            self.nb_workers = len(workers)

            # slow down this loop...
            sleep(dt_small)
            if self.nb_workers_cpu >= nb_max_workers:
                logger.debug(
                    cstring(
                        ("The workers are saturated: " "{}, sleep {} s").format(
                            self.nb_workers_cpu, dt
                        ),
                        color="WARNING",
                    )
                )
                sleep(dt)

            for q in self.queues:
                if not q.is_empty():
                    logger.debug(q)
                    logger.debug("check_and_act for work: " + repr(q.work))
                    try:
                        new_workers = q.check_and_act(sequential=sequential)
                    except OSError:
                        logger.error(
                            cstring(
                                "Memory full: to free some memory, no more "
                                "computing job will be launched while the last "
                                "(saving) waiting queue is not empty.",
                                color="FAIL",
                            )
                        )
                        log_memory_usage(color="FAIL", mode="error")
                        self._clear_save_queue(workers, sequential)
                        logger.info(
                            cstring(
                                "The last waiting queue has been emptied.",
                                color="FAIL",
                            )
                        )
                        log_memory_usage(color="FAIL", mode="info")
                        continue

                    if new_workers is not None:
                        for worker in new_workers:
                            workers.append(worker)
                    logger.debug("workers: " + repr(workers))

            if self.thread_check_works_t.exitcode:
                raise self.thread_check_works_t.exception

            if self.thread_check_works_p.exitcode:
                raise self.thread_check_works_p.exception

            if len(workers) != self.nb_workers:
                gc.collect()

        if self._has_to_stop:
            logger.info(
                cstring(
                    "Will exist because of signal 12.",
                    "Waiting for all workers to finish...",
                    color="FAIL",
                )
            )
            self._clear_save_queue(workers, sequential)

        self.thread_check_works_t.has_to_stop = True
        self.thread_check_works_p.has_to_stop = True
        self.thread_check_works_t.join()
        self.thread_check_works_p.join()

        self._print_at_exit(time() - self.t_start)
        log_memory_usage(time_as_str(2) + ": end of `compute`. mem usage")

        if self._has_to_stop and has_to_exit:
            logger.info(cstring("Exit with signal 99.", color="FAIL"))
            exit(99)

        self._reset_std_as_default()

    def _reset_std_as_default(self):
        sys.stdout = _stdout_at_import
        sys.stderr = _stderr_at_import
        self._log_file.close()

    def _clear_save_queue(self, workers, sequential):
        """Clear the last queue (which is often saving) before stopping."""
        q = self.queues[-1]

        idebug = 0
        # if the last queue is a WaitingQueueThreading (saving),
        # it is also emptied.
        while len(workers) > 0 or (
            not q.is_empty() and isinstance(q, WaitingQueueThreading)
        ):

            sleep(0.5)

            if len(workers) == 1 and q.is_empty():
                idebug += 1
                p = workers[0]
                if idebug == 100:
                    print("Issue:", p, p.exitcode)
            # from fluiddyn import ipydebug
            # ipydebug()

            if not q.is_empty() and isinstance(q, WaitingQueueThreading):
                new_workers = q.check_and_act(sequential=sequential)
                if new_workers is not None:
                    for worker in new_workers:
                        workers.append(worker)

    # workers[:] = [w for w in workers
    #               if not w.fill_destination()]

    def _print_at_exit(self, time_since_start):
        """Print information before exit."""
        txt = "Stop compute after t = {:.2f} s".format(time_since_start)
        try:
            nb_results = len(self.results)
        except AttributeError:
            nb_results = None
        if nb_results is not None and nb_results > 0:
            txt += " ({} results, {:.2f} s/result).".format(
                nb_results, time_since_start / nb_results
            )
        else:
            txt += "."

        if hasattr(self, "path_dir_result"):
            txt += "\npath results:\n" + self.path_dir_result

        print(txt)

    def make_code_graphviz(self, name_file):
        """Generate the graphviz / dot code."""

        code = "digraph {\nrankdir = LR\ncompound=true\n"
        # waiting queues
        code += '\nnode [shape="record"]\n'
        txt_queue = (
            '{name_quoted:40s} [label="<f0> {name}|'
            + "|".join(["<f{}>".format(i) for i in range(1, 5)])
            + '"]\n'
        )

        for q in self.queues:
            name_quoted = '"{}"'.format(q.name)
            code += txt_queue.format(name=q.name, name_quoted=name_quoted)

        # works and links
        code += '\nnode [shape="ellipse"]\n'

        txt_work = '{name_quoted:40s} [label="{name}"]\n'
        for q in self.queues:
            name_work = q.work_name or str(q.work)
            name_quoted = '"{}"'.format(name_work)
            code += txt_work.format(name=name_work, name_quoted=name_quoted)

        code += "\n"

        for q in self.queues:
            name_work = q.work_name or str(q.work)
            code += '{:40s} -> "{}"\n'.format('"' + q.name + '"', name_work)
            if hasattr(q.destination, "name"):
                code += '{:40s} -> "{}"\n'.format(
                    '"' + name_work + '"', q.destination.name
                )

        code += "}\n"

        with open(name_file, "w") as f:
            f.write(code)

        print(
            "A graph can be produced with one of these commands:\n"
            "dot topo.dot -Tpng -o topo.png\n"
            "dot topo.dot -Tx11"
        )
