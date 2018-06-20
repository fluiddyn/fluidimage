"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio

from fluidimage.util.util import logger
from fluidimage.topologies.experimental.executer_base import ExecuterBase
from fluidimage.topologies.experimental.nb_workers import nb_max_workers


class ExecuterAwaitMultiprocs(ExecuterBase):

    def __init__(self, topology):
        super().__init__(topology)
        #fonction container
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        self.define_function()


        print("\nWhat's in function dicts ?\n ")
        for key, af in self.async_funcs.items():
            print("async func : {} ".format(key))
        for key, f in self.funcs.items():
            print("func : {} ".format(key))
        print("\n")
        self.do_one_shot_job()
        trio.run(self.start_async)

        #ascyncio loop

    async def process(self):
        for key, af in reversed(self.async_funcs.items()):
            await af()

    def define_function(self):
        #define functions and store them un self.funcs
        for w in reversed(self.topology.works):
            print(w.name)
            if w.kind is not None and "one shot" in w.kind:
                def func(work=w, *args, **kwargs):
                    print("funtion {} is called".format(work.name))
                    work.func_or_cls(work.input_queue, work.output_queue)
                self.funcs[w.name] = func
            else:
                async def func(work=w, *args, **kwargs):
                    print("funtion {} is called".format(work.name))
                    work.func_or_cls(work.input_queue, work.output_queue)
                self.async_funcs[w.name] = func

    def compute(self, sequential = True, has_to_exit= False):
        print("compute")

    def do_one_shot_job(self):
        for key, func in reversed(self.funcs.items()):
            logger.info("Does one_shot_job key func : {} with function {}".format(key, func))
            func()

    async def start_async(self):
        """
        Define a concurenced work which is destined to be compute in a single process
        :param listdir: list of image names to compute
        :type list
        :return:
            """
        tasks = []
        async with trio.open_nursery() as nursery:
            for _ in range(40):
                tasks.append(
                    nursery.start_soon(self.process))
