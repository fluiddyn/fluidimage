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

    async def process(self):
        for key, af in reversed(self.async_funcs.items()):
            await af()

    def define_function(self):
        #define functions and store them
        for w in reversed(self.topology.works):
            print(w.name)
            if w.kind is not None and "one shot" in w.kind: ##One shot functionw
                def func(work=w):
                    print("funtion {} is called".format(work.name))
                    work.func_or_cls(work.input_queue, work.output_queue)
                self.funcs[w.name] = func
                continue
            elif w.kind is not None and "global" in w.kind:  ## global functions
                async def func(work=w):
                    print("global funtion {} is called".format(work.name))
                    work.func_or_cls(work.input_queue, work.output_queue)
            elif w.output_queue is not None: ### other function
                async def func(work=w):
                    print("funtion {} is called".format(work.name))
                    if work.input_queue.queue:
                        key, obj = await work.input_queue.queue.popitem()
                        ret =  work.func_or_cls(obj)
                        work.output_queue.queue[key] = ret
            else: #Last work
                async def func(work=w):
                    print("funtion {} is called".format(work.name))
                    if work.input_queue.queue:
                        key, obj = await work.input_queue.queue.popitem()
                        work.func_or_cls(obj)
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

    async def start_async_works(self):
        async with trio.open_nursery() as nursery:
            for f in self.async_funcs :
                nursery.start_soon(f)

        logger.info("Work all done in {}".format(time.time() - self.t_start))