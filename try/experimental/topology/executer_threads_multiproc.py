"""executer_threads_multiproc

An executer with the same methods as the "standard" fluidimage topology,
i.e. using threads for IO bounded tasks and multiprocessing for CPU-bounded tasks.

"""


from time import sleep, time
import gc
import os
import inspect
import scipy.io
import threading
from multiprocessing import Process

try:
    import queue
except ImportError:
    # python 2
    import Queue as queue


from fluiddyn import time_as_str


from ...util.util import cstring, logger, log_memory_usage
from ..waiting_queues.base import (
    WaitingQueueMultiprocessing,
    WaitingQueueThreading,
    WaitingQueueMakeCouple,
    WaitingQueueLoadImage,
    WaitingQueueBase,
    WaitingQueueOneShot,
)
from fluidimage.topologies.experimental.base import Queue
from fluidimage.topologies.experimental.base import Work
from .executer_base import ExecuterBase
from .nb_workers import nb_max_workers

dt = 0.25  # s
dt_small = 0.02
dt_update = 0.1


class ExecuterThreadsMultiprocs(ExecuterBase):
    def __init__(self, topology):
        super().__init__(topology)
        # we have to create a list of queues (WaitingQueueMultiprocessing,
        # WaitingQueueThreading, WaitingQueueMakeCouple, WaitingQueueLoadImage)
        # from the topology...

        # create waiting queues :
        # for q in reversed(topology.queues):
        #         self.topology.queues.append(WaitingQueueBase(name=q.name, work=q.name))

        # implement topologie
        self.add_queues()

        for q in self.topology.queues:
            # print(q.name)
            print(type(q))

        self._has_to_stop = False

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

        self.start_works()

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
                super().__init__()
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

                super().in_time_loop()

        self.thread_check_works_t = CheckWorksThread()
        self.thread_check_works_t.start()

        self.thread_check_works_p = CheckWorksProcess()
        self.thread_check_works_p.start()

        # self.topology.queues[1].destination = self.topology.queues[4]
        # self.topology.queues[3].destination = self.topology.queues[4]
        self.topology.queues[2].destination = self.topology.queues[3]
        self.topology.queues[4].destination = self.topology.queues[5]

        while not self._has_to_stop and (
            any([not q.is_empty() for q in self.topology.queues])
            or len(workers) > 0
        ):
            print("###WHILE###")
            for q in self.topology.queues:
                print(
                    "{}  : {}, {}, {}".format(
                        len(q), q.name, type(q), q.destination
                    )
                )
            # debug
            # if logger.level == 10 and \
            #    all([q.is_empty() for q in self.topology.queues]) and len(workers) == 1:
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

            for q in self.topology.queues:
                if not q.is_empty() and not isinstance(q, WaitingQueueOneShot):
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

        # TODO self._print_at_exit(time() - self.t_start)

        log_memory_usage(time_as_str(2) + ": end of `compute`. mem usage")

        if self._has_to_stop and has_to_exit:
            logger.info(cstring("Exit with signal 99.", color="FAIL"))
            exit(99)

    def add_queues(self):
        """
        fill self.topology.queues with appropriate queues from waiting_queue_base
        change work.input_queue and work.output_queue with previous queues ( from waiting_queue_base )
        Considere tuples ( tuple are changed in list for simplicity )
        :return:
        """
        # TODO considered that first and last queue can be tuple
        # TODO Simplify the code
        for work in reversed(self.topology.works):
            if work.input_queue is not None:  # First work or no input queue
                if isinstance(
                    work.input_queue, tuple
                ):  # Tuple : Many input_queue
                    for q in work.input_queue:
                        destination = self.give_destination(work)
                        new_queue = self.give_correspondant_waiting_queue(
                            work, destination, q
                        )
                        self.replace_queue(new_queue)
                        # attribute output queue
                        for q in self.topology.queues:
                            if work.output_queue is not None and not isinstance(
                                work.output_queue, tuple
                            ):
                                if work.output_queue.__dict__["name"] is q.name:
                                    work.output_queue = q

                else:  # One input queue
                    destination = self.give_destination(work)
                    new_queue = self.give_correspondant_waiting_queue(
                        work, destination
                    )
                    self.replace_queue(new_queue)
                    # attribute output queue

                    if work.output_queue is not None and not isinstance(
                        work.output_queue, tuple
                    ):
                        for q in self.topology.queues:
                            if work.output_queue.__dict__["name"] is q.name:
                                work.output_queue = q
                    elif isinstance(work.output_queue, tuple):
                        lst_q = []

                        for iq in work.output_queue:
                            for q in self.topology.queues:
                                if iq.__dict__["name"] is q.name:
                                    lst_q.append(q)
                        work.output_queue = lst_q

            else:  # fisrt work : attribute output queue
                for q in self.topology.queues:
                    if work.output_queue.__dict__["name"] is q.name:
                        work.output_queue = q
        # attribute input queue
        for w in self.topology.works:
            if isinstance(w.input_queue, tuple):
                lst_q = []
                for iq in w.input_queue:
                    for q in self.topology.queues:
                        if iq.__dict__["name"] is q.name:
                            lst_q.append(q)
                w.input_queue = lst_q
            else:
                for q in self.topology.queues:
                    if w.input_queue is not None:
                        if w.input_queue.__dict__["name"] is q.name:
                            w.input_queue = q

        for w in self.topology.works:
            print(
                "###WORK### name = {}, kind = {} input queue = {}, output_queue = {}".format(
                    w.name, w.kind, type(w.input_queue), type(w.output_queue)
                )
            )

    def give_destination(self, work):
        if len(self.topology.queues) is 0:
            return None
        for q in self.topology.queues:
            if work.output_queue is q.name:
                return q

    def give_correspondant_waiting_queue(
        self, work, destination, input_queue=None
    ):
        if input_queue is None:
            input_queue = work.input_queue
        queue_name = input_queue.__dict__["name"]
        #
        if input_queue is None:  # No input queue
            print(f"Work {input_queue} has no input queue")
        elif work.kind is not None:  # kind is not empty
            if "one shot" in work.kind:
                print(f"WORK func_or_cls {work}")
                queue = WaitingQueueOneShot(
                    name=input_queue.name,
                    work_name=work.name,
                    work=work.func_or_cls,
                    destination=destination,
                    topology=self.topology,
                )

            elif "io" in work.kind:
                queue = WaitingQueueMultiprocessing(
                    name=input_queue.name,
                    work_name=work.name,
                    work=work.func_or_cls,
                    destination=destination,
                    topology=self.topology,
                )
            else:
                if "global" in work.kind:
                    queue = WaitingQueueMultiprocessing(
                        name=queue_name,
                        work=work.func_or_cls,
                        destination=destination,
                        topology=self.topology,
                    )
                else:
                    queue = WaitingQueueMultiprocessing(
                        name=queue_name,
                        work=work.func_or_cls,
                        destination=destination,
                        topology=self.topology,
                    )
        else:
            queue = WaitingQueueMultiprocessing(
                name=queue_name,
                work_name=str(work.func_or_cls),
                work=work.func_or_cls,
                destination=destination,
                topology=self.topology,
            )
        return queue

    def start_works(self):
        for w in self.topology.works:
            print(
                "###WORK### name = {}, kind = {} input queue = {}, output_queue = {}".format(
                    w.name, w.kind, type(w.input_queue), type(w.output_queue)
                )
            )
            if w.input_queue is None:  # First work or no queue before work
                w.func_or_cls(w.input_queue, w.output_queue)
            elif w.kind is not None and "one shot" in w.kind:
                w.func_or_cls(w.input_queue, w.output_queue)
            # elif w.output_queue is None:  # First work or no queue before work ( SAVE )
            #     key, obj = w.input_queue.popitem()
            #     # TODO use dataobject save method
            #     path_save = '../../../../resultsPIVTMP'
            #     scipy.io.savemat(
            #         path_save,
            #         mdict={
            #             "deltaxs": obj.deltaxs,
            #             "deltays": obj.deltays,
            #             "xs": obj.xs,
            #             "ys": obj.ys,
            #         },
            #     )
            # elif w.kind is not None:  # kind is not empty
            #     if "global" not in w.kind:
            #         if inspect.isclass(w.func_or_cls):  # it's a class
            #             queue = w.input_queue
            #             w.func_or_cls(queue, w.output_queue)
            #         elif inspect.isfunction(w.func_or_cls):  # its a function
            #             key, obj = w.input_queue.popitem()
            #             ret = w.func_or_cls(obj)
            #             w.output_queue[key] = ret
            #         else:
            #             logger.warn("Neither a class nor a function")
            #     elif "global" in w.kind:
            #         w.func_or_cls(w.input_queue, w.output_queue)
            # else:  # No kind attribute
            #     key, obj = w.input_queue.popitem()
            #     ret = w.func_or_cls(obj)
            #     lighPiv = ret.make_light_result()
            #     w.output_queue[key] = lighPiv

    def replace_queue(self, queue):
        for index, q in enumerate(self.topology.queues):
            if q.name is queue.name:
                self.topology.queues[index] = queue
                return

    def _clear_save_queue(self, workers, sequential):
        """Clear the last queue (which is often saving) before stopping."""
        q = self.topology.queues[-1]

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
