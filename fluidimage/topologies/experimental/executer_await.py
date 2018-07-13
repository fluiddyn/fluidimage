"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio
import time
from fluiddyn import time_as_str
from fluidimage.util.util import logger, get_memory_usage
from fluidimage.topologies.experimental.executer_base import ExecuterBase
from multiprocessing import Process
from itertools import islice


class Executer(ExecuterBase):
    def __init__(self, topology, worker_limit = 6, queues_limit = 10, sleep_time=0.1,  new_dict=False):
        super().__init__(topology)
        if new_dict is not False:
            topology.queues[0].queue = new_dict
        self.t_start = time.time()
        # fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        self.sleep_time = sleep_time
        # server
        self.nb_working_worker = 0
        self.worker_limit = worker_limit
        self.server = None
        # #fonctions definition
        self.get_async_works()
        self.define_function()
        # Queue
        self.queues_limit = queues_limit

    def compute(self, sequential=True, has_to_exit=False):
        """
        Compute the topology
        :param sequential:
        :param has_to_exit:
        :return:
        """

        self.t_start = time.time()
        self.do_one_shot_job()
        self.print_queues()
        trio.run(self.start_async_works)

    async def start_async_works(self,):
        # Start soon all async functions
        async with trio.open_nursery() as self.nursery:
            for key, af in reversed(self.async_funcs.items()):
                self.nursery.start_soon(af)
        return

    def define_function(self):
        # defines functions and stores them
        for w in reversed(self.topology.works):
            # One shot functions
            if w.kind is not None and "one shot" in w.kind:

                def func(work=w):
                    work.func_or_cls(work.input_queue, work.output_queue)
                self.funcs[w.name] = func
                continue

            # global functions
            elif w.kind is not None and "global" in w.kind:

                async def func(work=w):
                    while True:
                        while len(work.output_queue.queue) > self.queues_limit:
                            await trio.sleep(self.sleep_time )
                        t_start = time.time()
                        while not work.func_or_cls(work.input_queue, work.output_queue):
                            if self.has_to_stop() and self.nb_working_worker == 0:
                                return
                            await trio.sleep(self.sleep_time)
                            t_start = time.time()
                        logger.info(
                            "[92m{:.3f} s. Launch work {} {} . mem usage: {:.3f} Mb[0m".format(
                                time.time() - self.t_start,
                                work.name.replace(" ", "_"),
                                "couple",
                                get_memory_usage(),
                            )
                        )
                        logger.info(
                            "work {} {} done in {:.3f} s".format(
                                work.name.replace(" ", "_"), "couple", time.time() - t_start
                            )
                        )
                        await trio.sleep(self.sleep_time)
                        self.print_queues()

            # I/O
            elif w.kind is not None and "io" in w.kind and w.output_queue is not None:

                async def func(work=w):
                    while True:
                        while (
                                not work.input_queue.queue
                                or len(work.output_queue.queue) > self.queues_limit
                        ):
                            if self.has_to_stop() and self.nb_working_worker == 0:
                                return
                            await trio.sleep(self.sleep_time)
                        start = time.time()
                        work.input_queue.queue["in_working"] = True
                        key, obj = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        logger.info(
                            "[92m{:.3f} s. Launch work {} {} . mem usage: {:.3f} Mb[0m".format(
                                time.time() - self.t_start,
                                work.name.replace(" ", "_"),
                                key,
                                get_memory_usage(),
                            )
                        )
                        ret = await trio.run_sync_in_worker_thread(
                            work.func_or_cls, obj
                        )
                        await trio.run_sync_in_worker_thread(
                            self.push, key, ret, work.output_queue.queue
                        )
                        self.print_queues()
                        del work.input_queue.queue["in_working"]
                        logger.info(
                            "work {} {} done in {:.3f} s".format(
                                work.name.replace(" ", "_"), key, time.time() - start
                            )
                        )
                        await trio.sleep(self.sleep_time)

            #there is output_queue
            elif w.output_queue is not None:
                async def func(q_c2s=None, q_s2c=None, work=w):
                    while True:
                        while (
                                not work.input_queue.queue
                                or self.nb_working_worker >= self.worker_limit
                                or len(work.output_queue.queue) > self.queues_limit
                        ):
                            if self.has_to_stop() and self.nb_working_worker == 0:
                                return
                            await trio.sleep(self.sleep_time)
                        start = time.time()
                        work.input_queue.queue["in_working"] = True
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        self.nursery.start_soon(self.worker, work, key, obj)
                        del work.input_queue.queue["in_working"]
                        self.print_queues()
                        await trio.sleep(self.sleep_time)
            #There is no output_queue
            else:
                async def func(q_c2s=None, q_s2c=None, work=w):
                    while True:
                        while (
                                not work.input_queue.queue
                                or self.nb_working_worker >= self.worker_limit
                        ):
                            if self.has_to_stop() and self.nb_working_worker == 0:
                                return
                            await trio.sleep(self.sleep_time)
                        start = time.time()
                        work.input_queue.queue["in_working"] = True
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        self.nursery.start_soon(self.worker, work, key, obj)
                        del work.input_queue.queue["in_working"]
                        self.print_queues()
                        await trio.sleep(self.sleep_time)
            self.async_funcs[w.name] = func

    def do_one_shot_job(self):
        for key, func in reversed(self.funcs.items()):
            logger.info(
                "Does one_shot_job key func : {} with function {}".format(key, func)
            )
            func()

    async def worker(self, work, key, obj):
        self.nb_working_worker += 1
        print("#### nb Worker", self.nb_working_worker)
        t_start = time.time()
        logger.info(
            "[92m{:.3f} s. Launch work {} {}. mem usage: {:.3f} Mb[0m".format(
                time.time() - self.t_start,
                work.name.replace(" ", "_"),
                key,
                get_memory_usage(),
            )
        )
        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
        if work.output_queue is not None:
            self.push(key, obj, work.output_queue.queue)
            work.output_queue.queue[key] = ret
        logger.info(
            "work {} {} done in {:.3f} s".format(
                work.name.replace(" ", "_"), key, time.time() - t_start
            )
        )
        self.nb_working_worker -= 1
        return

    def get_async_works(self):
        """
        Picks up async works and stores them in self.works
        :return:
        """
        for w in self.topology.works:
            if w.kind is None or "one shot" not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        """
        Work has to stop flag
        :return:
        """
        return not any([len(q.queue) != 0 for q in self.topology.queues])

    def pull(self, input_queue):
        key, obj = input_queue.popitem()
        if key is "in_working":
            key, obj = input_queue.popitem()
            input_queue["in_working"] = True
        return key, obj

    def push(self, key, obj, output_queue):
        output_queue[key] = obj
        return

    def print_queues(self):
        for q in self.topology.queues:
            print("{} : {} ".format(len(q.queue), q.name))
        print("\n")


class ExecuterAwaitMultiprocs(ExecuterBase):
    def __init__(self, topology, multi_executor=False):
        super().__init__(topology)
        self.multi_executor = multi_executor
        print(len(topology.queues[0].queue))

    def compute(self, sequential=True, has_to_exit=False):
        """
        Compute the topology
        :param sequential:
        :param has_to_exit:
        :return:
        """
        logger.info(
            "[92m{}: start compute. mem usage: {} Mb[0m".format(time_as_str(2), get_memory_usage())
        )
        self.t_start = time.time()
        if self.multi_executor is True:
            self.multi_executor_compute(nb_process=4)
        else:
            executor_await = Executer(self.topology)
            executor_await.do_one_shot_job()
            executor_await.print_queues()
            trio.run(executor_await.start_async_works)
        logger.info(
            "[92m{}: end of `compute`. mem usage {} Mb[0m".format(time_as_str(2), get_memory_usage())
        )
        print("Work all done in {:.5f} s".format(time.time() - self.t_start))

    def multi_executor_compute(self, nb_process):
        # get the first queue
        print("start getting fist queue")
        for w in self.topology.works:
            if w.input_queue == None:
                first_queue = w.output_queue
                work_first_queue = w
                break
        # fill the first queue
        if isinstance(work_first_queue.output_queue, tuple):
            raise NotImplementedError('first work have two or more output_queues')
        work_first_queue.func_or_cls(input_queue=None, output_queue=first_queue)
        # split it
        dict_list = []
        for item in self.partion_dict(first_queue.queue, nb_process):
            dict_list.append(item)
        # change topology
        new_topology = self.topology
        del new_topology.works[0]
        # make process and executor
        processes = []
        for i in range(nb_process):
            print(dict_list[i])
            executor = Executer(
                new_topology, worker_limit=1, new_dict=dict_list[i]
            )
            p = Process(target=executor.compute)
            processes.append(p)
        for p in processes:
            p.start()

    def partion_dict(self, dict, num):
        slice = int(len(dict) / num)
        it = iter(dict)
        for i in range(0, len(dict), slice):
            yield {k: dict[k] for k in islice(it, slice)}
