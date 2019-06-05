"""Executor async/await (:mod:`fluidimage.executors.exec_async`)
================================================================

This executer uses await/async with trio library to put topology tasks in
concurrency.

A single executor (in one process) is used. If CPU bounded tasks are limited by
the Python GIL, the threads won't use at the same time the CPU.

This means that the work will run on one CPU at a time, except if the topology
uses compiled code releasing the GIL. In this case, the GIL can be bypassed and
computation can use many CPU at a time.

.. autoclass:: ExecutorAsync
   :members:
   :private-members:

"""

import signal
import time
from collections import OrderedDict

import trio

from fluidimage.util import cstring, log_debug, log_memory_usage, logger

from .base import ExecutorBase


class ExecutorAsync(ExecutorBase):
    """Executor async/await.

    The work in performed in a single process.

    """

    def __init__(
        self,
        topology,
        path_dir_result,
        nb_max_workers=None,
        nb_items_queue_max=None,
        sleep_time=0.01,
        logging_level="info",
        stop_if_error=False,
    ):
        super().__init__(
            topology,
            path_dir_result,
            nb_max_workers,
            nb_items_queue_max,
            logging_level=logging_level,
            stop_if_error=stop_if_error,
        )

        self.nb_working_workers_cpu = 0
        self.nb_working_workers_io = 0

        # Executor parameters
        self.sleep_time = sleep_time

        # function containers
        self.async_funcs = OrderedDict()
        self.funcs = OrderedDict()

        # Functions definition
        self.define_functions()

        def signal_handler(sig, frame):
            del sig, frame  # unused
            logger.info("Ctrl+C signal received...")
            self._has_to_stop = True
            self.nursery.cancel_scope.cancel()
            # it seems that we don't need to raise the exception
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, signal_handler)

        # to avoid a pylint warning
        self.nursery = None

    def compute(self):
        """Compute the whole topology.

        Begin by executing one shot jobs, then execute multiple shots jobs
        implemented as async functions.  Warning, one shot jobs must be ancestors
        of multiple shots jobs in the topology.

        """

        self._init_compute()
        self.exec_one_shot_works()
        trio.run(self.start_async_works)
        self._finalize_compute()

    async def start_async_works(self):
        """Create a trio nursery and start all async functions.

        """
        async with trio.open_nursery() as self.nursery:
            for af in self.async_funcs.values():
                self.nursery.start_soon(af)

            self.nursery.start_soon(self.update_has_to_stop)

    def define_functions(self):
        """Define sync and async functions.

        Define sync ("one shot" functions) and async functions (multiple shot
        functions), and store them in `self.async_funcs`.

        The behavior of the executor is mostly defined here.  To sum up : Each
        "multiple shot" waits for an items to be available in there input_queue
        and process the items as soon as they are available.

        """
        for work in self.works:

            # global functions
            if work.kind is not None and "global" in work.kind:

                async def func(work=work):
                    while True:
                        while (
                            isinstance(work.input_queue, tuple)
                            and all(not q for q in work.input_queue)
                        ) or not work.input_queue:
                            await trio.sleep(self.sleep_time)
                            if self._has_to_stop:
                                return
                        t_start = time.time()
                        log_memory_usage(
                            f"{time.time() - self.t_start:.2f} s. Launch work "
                            + work.name_no_space
                            + f" (?). mem usage"
                        )
                        work.func_or_cls(work.input_queue, work.output_queue)
                        if self._has_to_stop:
                            return
                        await trio.sleep(self.sleep_time)

                        logger.info(
                            f"work {work.name_no_space} "
                            f"done in {time.time() - t_start:.3f} s"
                        )

                        await trio.sleep(self.sleep_time)

            # I/O
            elif work.kind is not None and (
                "io" in work.kind or work.kind == "io"
            ):
                func = self.def_async_func_work_io(work)

            # CPU-bounded work
            else:
                func = self.def_async_func_work_cpu(work)

            self.async_funcs[work.name] = func

    def def_async_func_work_io(self, work):
        """Define an asynchronous function launching a io work."""

        async def func(work=work):
            while True:
                while (
                    not work.input_queue
                    or self.nb_working_workers_io >= self.nb_max_workers
                    or (
                        work.output_queue is not None
                        and len(work.output_queue) >= self.nb_items_queue_max
                    )
                ):
                    if self._has_to_stop:
                        return
                    await trio.sleep(self.sleep_time)
                self.nursery.start_soon(self.async_run_work_io, work)
                await trio.sleep(self.sleep_time)

        return func

    def def_async_func_work_cpu(self, work):
        """Define an asynchronous function launching a cpu work."""

        async def func(work=work):
            while True:
                while (
                    not work.input_queue
                    or self.nb_working_workers_cpu >= self.nb_max_workers
                    or (
                        work.output_queue is not None
                        and len(work.output_queue) >= self.nb_items_queue_max
                    )
                ):
                    if self._has_to_stop:
                        return
                    await trio.sleep(self.sleep_time)
                self.nursery.start_soon(self.async_run_work_cpu, work)
                await trio.sleep(self.sleep_time)

        return func

    async def async_run_work_io(self, work):
        """Is destined to be started with a "trio.start_soon".

        Executes the work on an item (key, obj), and add the result on
        work.output_queue.

        Parameters
        ----------

        work :

          A work from the topology

        """
        self.nb_working_workers_io += 1

        try:
            key, obj = work.input_queue.pop_first_item()
        except KeyError:
            self.nb_working_workers_io -= 1
            return

        if work.check_exception(key, obj):
            self.nb_working_workers_io -= 1
            return

        t_start = time.time()
        log_memory_usage(
            f"{time.time() - self.t_start:.2f} s. Launch work "
            + work.name_no_space
            + f" ({key}). mem usage"
        )
        # pylint: disable=W0703
        try:
            ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        except Exception as error:
            self.log_exception(error, work.name_no_space, key)
            if self.stop_if_error:
                raise
            ret = error
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_io -= 1

    async def async_run_work_cpu(self, work):
        """Is destined to be started with a "trio.start_soon".

        Executes the work on an item (key, obj), and add the result on
        work.output_queue.

        Parameters
        ----------

        work :

          A work from the topology

        """
        self.nb_working_workers_cpu += 1

        try:
            key, obj = work.input_queue.pop_first_item()
        except KeyError:
            self.nb_working_workers_cpu -= 1
            return

        if work.check_exception(key, obj):
            self.nb_working_workers_cpu -= 1
            return

        t_start = time.time()
        log_memory_usage(
            f"{time.time() - self.t_start:.2f} s. Launch work "
            + work.name_no_space
            + f" ({key}). mem usage"
        )
        # pylint: disable=W0703
        try:
            ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        except Exception as error:
            self.log_exception(error, work.name_no_space, key)
            if self.stop_if_error:
                raise
            ret = error
        else:
            logger.info(
                f"work {work.name_no_space} ({key}) "
                f"done in {time.time() - t_start:.3f} s"
            )

        if work.output_queue is not None:
            work.output_queue[key] = ret
        self.nb_working_workers_cpu -= 1

    async def update_has_to_stop(self):
        """Work has to stop flag. Check if all works has been done.

        Return True if there are no workers in working and if there is no items in
        all queues.

        """

        while not self._has_to_stop:

            result = (
                (not any([bool(queue) for queue in self.topology.queues]))
                and self.nb_working_workers_cpu == 0
                and self.nb_working_workers_io == 0
            )

            if result:
                self._has_to_stop = True
                log_debug(f"has_to_stop!")

            if self.logging_level == "debug":
                log_debug(f"self.topology.queues: {self.topology.queues}")
                log_debug(
                    f"self.nb_working_workers_cpu: {self.nb_working_workers_cpu}"
                )
                log_debug(
                    f"self.nb_working_workers_io: {self.nb_working_workers_io}"
                )

            await trio.sleep(self.sleep_time)
