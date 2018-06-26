"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio
import time
from fluidimage.util.util import logger
from fluidimage.topologies.experimental.executer_base import ExecuterBase
from fluidimage.topologies.experimental.nb_workers import nb_max_workers


class ExecuterAwaitMultiprocs(ExecuterBase):

    def __init__(self, topology):
        super().__init__(topology)
        self.t_start = time.time()
        #fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        #fonctions definition
        self.get_async_works()
        self.define_function()
        print("\nWhat's in function dicts ?\n ")
        for key, af in self.async_funcs.items():
            print("async func : {} ".format(key))
        for key, f in self.funcs.items():
            print("func : {} ".format(key))
        print("\n")

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
                async def func(cond, work=w):
                    print("global funtion {} is called".format(work.name))
                    async with cond:
                        while not self.has_to_stop():
                            while not work.func_or_cls(work.input_queue, work.output_queue):
                                print("global funtion {} is whiling".format(work.name))
                                cond.notify()
                                await cond.wait()
                            cond.notify()
                            await cond.wait()
                        print("global funtion {} is have finished working".format(work.name))
            # I/O
            elif w.kind is not None and "io" in w.kind and w.output_queue is not None:
                async def func(cond, work=w):
                    async with cond:
                        while not self.has_to_stop():
                            print("funtion {} is called".format(work.name))
                            while not work.input_queue.queue:
                                print("global funtion {} is whiling".format(work.name))
                                cond.notify()
                                await cond.wait()
                            key, obj = work.input_queue.queue.popitem()
                            file = await trio.open_file(obj)
                            ret =  await trio.run_sync_in_worker_thread(work.func_or_cls,file)
                            work.output_queue.queue[key] = ret
                            cond.notify()
                            await cond.wait()
                        print("funtion {} is have finished working".format(work.name))
            # other functions that have an output queue
            elif w.output_queue is not None:
                async def func(cond, work=w):
                    async with cond:
                        while not self.has_to_stop():
                            print("funtion {} is called".format(work.name))
                            while not work.input_queue.queue:
                                print("global funtion {} is whiling".format(work.name))
                                cond.notify()
                                await cond.wait()
                            key, obj = work.input_queue.queue.popitem()
                            ret =  work.func_or_cls(obj)
                            work.output_queue.queue[key] = ret
                            cond.notify()
                            await cond.wait()
                        print("funtion {} is have finished working".format(work.name))
            else:
                # Last work : no output queue
                async def func(cond, work=w):
                    async with cond:
                        while not self.has_to_stop():
                            print("funtion {} is called".format(work.name))
                            while not work.input_queue.queue:
                                print("global funtion {} is whiling".format(work.name))
                                cond.notify()
                                await cond.wait()
                            key, obj = work.input_queue.queue.popitem()
                            await trio.run_sync_in_worker_thread(work.func_or_cls,obj)
                            cond.notify()
                            await cond.wait()
                        print("funtion {} is have finished working".format(work.name))

            self.async_funcs[w.name] = func

    def compute(self, sequential = True, has_to_exit= False):
        print("compute")
        self.do_one_shot_job()
        trio.run(self.start_async_works)


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

    async def start_async_works(self):
        async with trio.open_nursery() as nursery:
            cond = trio.Condition()
            for key, af in reversed(self.async_funcs.items()):
                nursery.start_soon(af, cond)

        logger.info("Work all done in {}".format(time.time() - self.t_start))


class multiproc:

    def __init__(self):
        pass


