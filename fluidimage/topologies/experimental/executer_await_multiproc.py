"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import asyncio

from fluidimage.util.util import logger
from .executer_base import ExecuterBase
from .nb_workers import nb_max_workers

class ExecuterAwaitMultiprocs(ExecuterBase):
    def __init__(self, topology):
        super().__init__(topology)
        #fonction container
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        self.loop = asyncio.get_event_loop()

        for w in reversed(self.topology.works):
            print(w.name)
            if w.kind is not None and "one shot" in w.kind:
                def func(name=w.name, *args, **kwargs):
                    print("funtion '{}' is called".format(name))
                    return w.func_or_cls(*args, **kwargs)
                self.funcs[w.name] = func
            else:
                async def func(name=w.name, *args, **kwargs):
                    print("async funtion {} is called".format(name))
                    return w.func_or_cls(*args, **kwargs)
                self.async_funcs[w.name] = func

        print("\nWhats in function dicts ?\n ")
        for key, af in self.async_funcs.items():
            print("async func : {} ".format(key))
        for key, f in self.funcs.items():
            print("func : {} ".format(key))
        print("\n")
        self.do_one_shot_job()
        # self.fill_queue()

        #ascyncio loop

    async def process(self, im1, im2, serie):
        logger.info("Now let's start process {} et {}".format(im1, im2))
        res = None
        for key, af in reversed(self.async_funcs.items()):
            print(key)



    def compute(self, sequential = True, has_to_exit= False):
        print("compute")

    def do_one_shot_job(self):
        for key,func in reversed(self.funcs.items()):
            logger.info("Does one_shot_job key func : {} with function {}".format(key, func))
            ret = func({}, {})
            print(ret)
            self.fill_queue(ret)
            break #TODO TOPOLOGIE non linear

    def fill_queue(self, serie):
        """
        Define a concurenced work which is destined to be compute in a single process
        :param listdir: list of image names to compute
        :type list
        :return:
        """
        tasks = []
        serie_names = serie.get_name_all_arrays()
        print(serie_names)
        for i in range(len(serie_names) - 1):
            a_serie = serie.get_next_serie()
            tasks.append(
                asyncio.ensure_future(self.process(serie_names[i], serie_names[i + 1], a_serie, ))
            )
        self.loop.run_until_complete(asyncio.wait(tasks))
        self.loop.close()

    def make_f(kwargs, k):
        def f():
            print(k, kwargs[k])
        return f