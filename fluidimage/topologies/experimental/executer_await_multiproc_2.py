"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio
import time
import rpyc
import aiohttp
from rpyc.utils.server import ThreadedServer
from rpyc import async as async_
from fluidimage.util.util import logger
from fluidimage.topologies.experimental.executer_base import ExecuterBase
from fluidimage.topologies.experimental.nb_workers import nb_max_workers
from .server_rpyc import MyClass


class ExecuterAwaitMultiprocs(ExecuterBase):

    def __init__(self, topology):
        super().__init__(topology)
        self.t_start = time.time()
        #fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        #server
        self.server = None
        trio.run(self.start_server)
        # #fonctions definition
        # self.get_async_works()
        # self.define_function()
        # print("\nWhat's in function dicts ?\n ")
        # for key, af in self.async_funcs.items():
        #     print("async func : {} ".format(key))
        # for key, f in self.funcs.items():
        #     print("func : {} ".format(key))
        # print("\n")


    async def process(self,cond):
        for key, af in reversed(self.async_funcs.items()):
            await af(cond)


    def define_function(self):
        #define functions and store them
        for w in reversed(self.topology.works):
            print(w.name)
            # One shot functions
            if w.kind is not None and "one shot" in w.kind:
                def func(work=w):
                    print("funtion {} is called".format(work.name))
                    work.func_or_cls(work.input_queue, work.output_queue)
                self.funcs[w.name] = func
                continue
            # global functions
            elif w.kind is not None and "global" in w.kind:
                async def func(work=w):
                    print("{} is called".format(work.name))
                    while True:
                        while not work.func_or_cls(work.input_queue, work.output_queue):
                            print("{} have nothing in his queue".format(work.name))
                            if self.has_to_stop():
                                return
                            await trio.sleep(1)
                        print("{} is working".format(work.name))
                        self.print_queues()
                        # I/O
            elif w.kind is not None and "io" in w.kind and w.output_queue is not None:
                async def func(work=w):
                    print("{} is called".format(work.name))
                    while True:
                        while not work.input_queue.queue:
                            print("{} have nothing in his queue".format(work.name))
                            if self.has_to_stop():
                                return
                            await trio.sleep(1)
                        self.print_queues()
                        print("{} is poping".format(work.name))
                        key, obj = await trio.run_sync_in_worker_thread(self.pull,work.input_queue.queue)
                        print("{} is working".format(work.name))
                        # ret = await trio.open_file(obj)
                        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
                        print("{} is pushing".format(work.name))
                        await trio.run_sync_in_worker_thread(self.push,key, ret,work.output_queue.queue)
                        self.print_queues()
            elif w.kind is not None and "server" in w.kind and w.output_queue is not None:
                async def func(work=w):
                    print("{} is called".format(work.name))
                    while True:
                        while not work.input_queue.queue:
                            print("{} have nothing in his queue".format(work.name))
                            if self.has_to_stop():
                                return
                            await trio.sleep(1)
                        self.print_queues()
                        print("{} is poping".format(work.name))
                        key, obj = await trio.run_sync_in_worker_thread(self.pull,work.input_queue.queue)
                        print("{} is working".format(work.name))
                        await self.send_work(work.func_or_cls, obj)

                        print("{} is pushing".format(work.name))
                        if work.output_queue is not None:
                            await trio.run_sync_in_worker_thread(self.push,key,ret,work.output_queue.queue)
                        self.print_queues()
            #other functions
            else:
                async def func(work=w):
                    print("{} is called".format(work.name))
                    while True:
                        while not work.input_queue.queue:
                            print("{} have nothing in his queue".format(work.name))
                            if self.has_to_stop():
                                return
                            await trio.sleep(1)
                        self.print_queues()
                        print("{} is poping".format(work.name))
                        (key, obj) = await trio.run_sync_in_worker_thread(self.pull,work.input_queue.queue)
                        print("{} is working".format(work.name))
                        ret = await trio.run_sync_in_worker_thread(work.func_or_cls, obj)
                        if work.output_queue is not None:
                            print("{} is pushing".format(work.name))
                            await trio.run_sync_in_worker_thread(self.push, key, ret, work.output_queue.queue)
                        self.print_queues()
            self.async_funcs[w.name] = func


    def compute(self, sequential = True, has_to_exit= False):
        print("compute")
        self.do_one_shot_job()
        trio.run(self.start_async_works,)#instruments=[Tracer()]

    def do_one_shot_job(self):
        for key, func in reversed(self.funcs.items()):
            logger.info("Does one_shot_job key func : {} with function {}".format(key, func))
            func()

    def get_async_works(self):
        for w in self.topology.works:
            if w.kind is None or 'one shot' not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        return not any([len(q.queue) != 0 for q in self.topology.queues])

    async def send_work(self,work,  obj):
        async with aiohttp.ClientSession() as session:
            async with session.get('localhost') as resp:
                print(resp.status)
                print(await
                resp.text())

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
        async with trio.open_nursery() as nursery:
            for key, af in reversed(self.async_funcs.items()):
                nursery.start_soon(af)
        logger.info("Work all done in {}".format(time.time() - self.t_start))




    def before_run(self):
        print("!!! run started")

    def _print_with_task(self, msg, task):
        # repr(task) is perhaps more useful than task.name in general,
        # but in context of a tutorial the extra noise is unhelpful.
        print("{}: {}".format(msg, task.name))

    def task_spawned(self, task):
        self._print_with_task("### new task spawned", task)

    def task_scheduled(self, task):
        self._print_with_task("### task scheduled", task)

    def before_task_step(self, task):
        self._print_with_task(">>> about to run one step of task", task)

    def after_task_step(self, task):
        self._print_with_task("<<< task step finished", task)

    def task_exited(self, task):
        self._print_with_task("### task exited", task)

    def before_io_wait(self, timeout):
        if timeout:
            print("### waiting for I/O for up to {} seconds".format(timeout))
        else:
            print("### doing a quick check for I/O")
        self._sleep_time = trio.current_time()

    def after_io_wait(self, timeout):
        duration = trio.current_time() - self._sleep_time
        print("### finished I/O check (took {} seconds)".format(duration))

    def after_run(self):
        print("!!! run finished")

