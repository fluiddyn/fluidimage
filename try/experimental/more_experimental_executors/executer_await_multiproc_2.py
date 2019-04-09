"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio
import time
import pickle
from fluidimage.util.util import logger
from fluidimage.experimental.executors.executor_base import ExecutorBase


class ExecuterAwaitMultiprocs(ExecutorBase):
    def __init__(self, topology):
        super().__init__(topology)
        self.t_start = time.time()
        # fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        self.sleep_time = 0.01
        # server
        self.server = None
        # #fonctions definition
        self.store_async_works()
        self.define_function()

    async def process(self, cond):
        for key, af in reversed(self.async_funcs.items()):
            await af(cond)

    def define_function(self):
        # define functions and store them
        for w in reversed(self.topology.works):
            print(w.name)
            # One shot functions
            if w.kind is not None and "one shot" in w.kind:

                def func(work=w):
                    print(f"funtion {work.name} is called")
                    work.func_or_cls(work.input_queue, work.output_queue)

                self.funcs[w.name] = func
                continue
            # global functions
            elif w.kind is not None and "global" in w.kind:

                async def func(work=w):
                    print(f"{work.name} is called")
                    while True:
                        while not work.func_or_cls(
                            work.input_queue, work.output_queue
                        ):
                            if self.has_to_stop() and not self.job_in_progress():
                                return
                            await trio.sleep(self.sleep_time)
                        await trio.sleep(self.sleep_time)
                        self.print_queues()

            # I/O
            elif (
                w.kind is not None
                and "io" in w.kind
                and w.output_queue is not None
            ):

                async def func(work=w):
                    print(f"{work.name} is called")
                    while True:
                        await trio.sleep(self.sleep_time)
                        while not work.input_queue.queue:
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        key, obj = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        work.input_queue.queue["test"] = True
                        ret = await trio.run_sync_in_worker_thread(
                            work.func_or_cls, obj
                        )
                        await trio.run_sync_in_worker_thread(
                            self.push, key, ret, work.output_queue.queue
                        )
                        del work.input_queue.queue["test"]
                        await trio.sleep(self.sleep_time)
                        self.print_queues()

            # server
            elif (
                w.kind is not None
                and "server" in w.kind
                and w.output_queue is not None
            ):

                async def func(work=w):
                    print(f"{work.name} is called")
                    while True:
                        await trio.sleep(self.sleep_time)
                        while not work.input_queue.queue:
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        key, obj = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        work.input_queue.queue["test"] = True
                        # server part
                        conn = await trio.open_tcp_stream("localhost", 8888)
                        obj_s = pickle.dumps(obj)
                        print("send obj")
                        await conn.send_all(obj_s)
                        await conn.send_eof()  # end sending signal
                        print("sent")
                        # receiving
                        data = []
                        while True:
                            packet = await conn.receive_some(4096)
                            print("packet")
                            if not packet:
                                break
                            data.append(packet)
                        print("end receiving")
                        try:
                            ret = pickle.loads(b"".join(data))
                            if work.output_queue is not None:
                                await trio.run_sync_in_worker_thread(
                                    self.push, key, ret, work.output_queue.queue
                                )
                            del work.input_queue.queue["test"]
                        except:
                            print("PICKEL ERROR ")
                        # other functions
                        self.print_queues()

            else:

                async def func(work=w):
                    print(f"{work.name} is called")
                    while True:
                        while not work.input_queue.queue:
                            if self.has_to_stop():
                                return
                            await trio.sleep(self.sleep_time)
                        (key, obj) = await trio.run_sync_in_worker_thread(
                            self.pull, work.input_queue.queue
                        )
                        work.input_queue.queue["test"] = True
                        ret = work.func_or_cls(obj)
                        if work.output_queue is not None:
                            await trio.run_sync_in_worker_thread(
                                self.push, key, ret, work.output_queue.queue
                            )
                        del work.input_queue.queue["test"]
                        await trio.sleep(self.sleep_time)
                        # put funcctions in async_foncs dict
                        print(self.print_queues())

            self.async_funcs[w.name] = func

    def compute(self, sequential=True, has_to_exit=False):
        print("compute")
        self.do_one_shot_job()
        trio.run(self.start_async_works)  # instruments=[Tracer()]

    def do_one_shot_job(self):
        for key, func in reversed(self.funcs.items()):
            logger.info(
                "Does one_shot_job key func : {} with function {}".format(
                    key, func
                )
            )
            func()

    def store_async_works(self):
        for w in self.topology.works:
            if w.kind is None or "one shot" not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        return not any([len(q.queue) != 0 for q in self.topology.queues])

    async def send_work(self, obj):
        conn = await trio.open_tcp_stream("localhost", 8888)
        obj_s = pickle.dumps(obj)
        await conn.send_all(obj_s)

    def pull(self, input_queue):
        key, obj = input_queue.popitem()
        return (key, obj)

    def push(self, key, obj, output_queue):
        output_queue[key] = obj
        return

    def print_queues(self):
        for q in self.topology.queues:
            print(len(q.queue))
        print("\n")

    async def start_async_works(self):
        async with trio.open_nursery() as self.nursery:
            for key, af in reversed(self.async_funcs.items()):
                self.nursery.start_soon(af)
        logger.info("Work all done in {}".format(time.time() - self.t_start))
