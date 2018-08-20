"""
Classic executor with multiprocessing and threading.
Not funtionnal yet !
"""


import os
from time import time, sleep
import threading
from multiprocessing import Process
import multiprocessing
from fluiddyn import time_as_str
from fluidimage.util.util import (
    logger,
    cstring,
    get_memory_usage,
    log_memory_usage,
)
from .executor_base import ExecutorBase

dt = 0.25  # s
dt_small = 0.02
dt_update = 0.1


class ExecutorClassic(ExecutorBase):
    def __init__(self, topology):
        super().__init__(topology)
        self.t_start = time()

    def compute(self, sequential=None, has_to_exit=True):

        self.do_one_shot_job()
        print("One shot jobs done")

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
                print("threads loop")
                t_tmp = time()
                for worker in workers:
                    if worker.finished:
                        print("#######removed worker {}".format(worker))
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
                    print(e)
                    self.exitcode = 1
                    self.exception = e

        class CheckWorksProcess(CheckWorksThread):
            cls_to_be_updated = Process

            def in_time_loop(self):
                # weird bug subprocessing py3
                print("process loop")
                for worker in workers:
                    if not worker.really_started:
                        worker.really_started = True
                        worker.launch_worker()

                        # print('check if worker has really started.' +
                        #       worker.key)
                        try:
                            worker.really_started = (
                                worker.comm_started.get_nowait()
                            )
                        except:  # TODO queue.Empty:
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
                            workers.remove(worker)

                super(CheckWorksProcess, self).in_time_loop()

        self.thread_check_works_t = CheckWorksThread()
        self.thread_check_works_t.start()

        self.thread_check_works_p = CheckWorksProcess()
        self.thread_check_works_p.start()

        # While all jobs are not finished
        item_number = 1
        while not self._has_to_stop():
            print("worker", workers)
            self.print_queue()
            self.nb_workers = len(workers)
            # slow down this loop...
            sleep(dt_small)
            if self.nb_workers_cpu >= self.nb_max_workers:
                logger.debug(
                    cstring(
                        ("The workers are saturated: " "{}, sleep {} s").format(
                            self.nb_workers_cpu, dt
                        ),
                        color="WARNING",
                    )
                )
                sleep(dt)

            # For all "multiple shot" works
            for work in self.topology.works:
                # TODO get multiple shot jobs in a list
                if work.kind is None or "one shot" not in work.kind:
                    # global function
                    if work.kind is not None and "global" in work.kind:
                        if len(workers) < self.nb_max_workers:
                            workers.append(
                                GlobalWorker(self.t_start, work, item_number)
                            )
                            item_number += 1
                    # I/O
                    elif (
                        work.kind is not None
                        and "io" in work.kind
                        and work.output_queue is not None
                    ):
                        if (
                            work.input_queue.queue
                            and len(workers) < self.nb_max_workers
                        ):

                            key, obj = work.input_queue.queue.popitem()
                            workers.append(Worker(self.t_start, work, key, obj))
                    # Other works
                    else:
                        if (
                            work.input_queue.queue
                            and len(workers) < self.nb_max_workers
                        ):
                            key, obj = work.input_queue.queue.popitem()
                            workers.append(Worker(self.t_start, work, key, obj))
                    sleep(dt)
        if self._has_to_stop:
            logger.info(
                cstring(
                    "Will exist because of signal 12.",
                    "Waiting for all workers to finish...",
                    color="FAIL",
                )
            )

        self.thread_check_works_t.has_to_stop = True
        self.thread_check_works_p.has_to_stop = True
        self.thread_check_works_t.join()
        self.thread_check_works_p.join()

        # self._print_at_exit(time() - self.t_start)
        log_memory_usage(time_as_str(2) + ": end of `compute`. mem usage")

    def do_one_shot_job(self):
        # For all one shot jobs
        for work in self.topology.works:
            if work.kind is not None and "one shot" in work.kind:
                # run function
                work.func_or_cls(work.input_queue, work.output_queue)

    def print_queue(self):
        for q in self.topology.queues:
            print(q.queue)

    def _has_to_stop(self):
        print(
            "has to stop",
            not any([len(q.queue) != 0 for q in self.topology.queues]),
        )
        return not any([len(q.queue) != 0 for q in self.topology.queues])


class Worker:
    @staticmethod
    def _Queue(*args, **kwargs):
        return multiprocessing.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return multiprocessing.Process(*args, **kwargs)

    def __init__(self, t_start, work, key, obj=None):
        self.work = work
        self.key = key
        self.obj = obj
        self.really_started = False
        self.finished = False
        self.t_start = t_start
        self.p = None

    def launch_worker(self):
        comm_started = self._Queue()
        self.p = self._Process(target=self.job)
        self.p.t_start = time()
        self.p.work_name = self.work.name
        self.p.comm_started = comm_started
        self.p.really_started = False
        self.p.start()
        self.p.key = self.key

    def terminate(self):
        self.p.terminate()

    def job(self):
        t_start = time()
        log_memory_usage(
            "{:.2f} s. ".format(time() - self.t_start)
            + "Launch work "
            + self.work.name.replace(" ", "_")
            + " ({}). mem usage".format(self.key)
        )
        ret = self.work.func_or_cls(self.obj)
        if self.work.output_queue is not None:
            print("work {} filling output_queue".format(self.work.name))
            self.work.output_queue.queue[self.key] = ret
        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                self.work.name.replace(" ", "_"), self.key, time() - t_start
            )
        )
        # TODO self._nb_workers -= 1
        self.finished = True


class GlobalWorker(Worker):
    def __init__(self, t_start, work, key, obj=None):
        super().__init__(t_start, work, key)

    def job(self):
        t_start = time()
        log_memory_usage(
            "{:.2f} s. ".format(time() - self.t_start)
            + "Launch work "
            + self.work.name.replace(" ", "_")
            + " ({}). mem usage".format(self.key)
        )
        while not self.work.func_or_cls(
            self.work.input_queue, self.work.output_queue
        ):
            sleep(dt_small)
        logger.info(
            "work {} ({}) done in {:.3f} s".format(
                self.work.name.replace(" ", "_"), self.key, time() - t_start
            )
        )
        # TODO self.nb_working_worker -= 1
        self.finished = True
