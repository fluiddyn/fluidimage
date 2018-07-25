"""
This executer splits the work in nb_max_workers multiprocessing workers.

IO tasks are handled with an asyncio event loops.

"""
import collections
import trio
import time
import rpyc
from rpyc.utils.server import ThreadedServer
from fluidimage.util.util import logger
from fluidimage.experimental.executors.executor_base import ExecutorBase
from .server_rpyc import MyClass


class ExecuterAwaitMultiprocs(ExecutorBase):
    def __init__(self, topology):
        super().__init__(topology)
        self.t_start = time.time()
        # fonction containers
        self.works = []
        self.async_funcs = collections.OrderedDict()
        self.funcs = collections.OrderedDict()
        # server
        self.server = None
        # self.start_server()
        # fonctions definition
        self.get_async_works()
        self.define_function()
        print("\nWhat's in function dicts ?\n ")
        for key, af in self.async_funcs.items():
            print("async func : {} ".format(key))
        for key, f in self.funcs.items():
            print("func : {} ".format(key))
        print("\n")

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
                            while not work.func_or_cls(
                                work.input_queue, work.output_queue
                            ):
                                cond.notify_all()
                                if self.has_to_stop():
                                    return
                                print(
                                    "global funtion {} is waiting".format(
                                        work.name
                                    )
                                )
                                await cond.wait()
                                print(
                                    "global funtion {} is waking up".format(
                                        work.name
                                    )
                                )
                            print(
                                "global funtion {} is working".format(work.name)
                            )
                            cond.notify_all()
                            print(
                                "global funtion {} is waiting".format(work.name)
                            )
                            await cond.wait()
                            print(
                                "global funtion {} is waking up".format(work.name)
                            )
                        print(
                            "global funtion {} is have finished working".format(
                                work.name
                            )
                        )

            # I/O
            elif (
                w.kind is not None
                and "io" in w.kind
                and w.output_queue is not None
            ):

                async def func(cond, work=w):
                    print("funtion {} is called".format(work.name))
                    async with cond:
                        while not self.has_to_stop():
                            while not work.input_queue.queue:
                                if self.has_to_stop():
                                    return
                                cond.notify_all()
                                print(
                                    "global funtion {} is waiting".format(
                                        work.name
                                    )
                                )
                                await cond.wait()
                                print(
                                    "global funtion {} is waking up".format(
                                        work.name
                                    )
                                )
                            print(
                                "global funtion {} is working".format(work.name)
                            )
                            key, obj = work.input_queue.queue.popitem()
                            # ret = await trio.open_file(obj)
                            ret = await trio.run_sync_in_worker_thread(
                                work.func_or_cls, obj
                            )
                            work.output_queue.queue[key] = ret
                            cond.notify_all()
                            print(
                                "global funtion {} is waiting".format(work.name)
                            )
                            await cond.wait()
                            print(
                                "global funtion {} is waking up".format(work.name)
                            )
                        print(
                            "funtion {} is have finished working".format(
                                work.name
                            )
                        )

            # other functions
            else:

                async def func(cond, work=w):
                    async with cond:
                        print("funtion {} is called".format(work.name))
                        while not self.has_to_stop():
                            while not work.input_queue.queue:
                                cond.notify_all()
                                if self.has_to_stop():
                                    return
                                print(
                                    "global funtion {} is waiting".format(
                                        work.name
                                    )
                                )
                                await cond.wait()
                                print(
                                    "global funtion {} is waking up".format(
                                        work.name
                                    )
                                )
                            print(
                                "global funtion {} is working".format(work.name)
                            )
                            key, obj = work.input_queue.queue.popitem()
                            conn = await trio.open_tcp_stream(
                                "localhost", port=18813
                            )
                            res = await trio.run_sync_in_worker_thread(
                                work.func_or_cls, obj
                            )
                            if work.output_queue is not None:
                                work.output_queue.queue[key] = res
                            cond.notify_all()
                            print(
                                "global funtion {} is waiting".format(work.name)
                            )
                            await cond.wait()
                            print(
                                "global funtion {} is waking up".format(work.name)
                            )
                        print(
                            "funtion {} is have finished working".format(
                                work.name
                            )
                        )

            self.async_funcs[w.name] = func

    def compute(self, sequential=True, has_to_exit=False):
        print("compute")
        self.do_one_shot_job()
        trio.run(self.start_async_works)

    def do_one_shot_job(self):
        for key, func in reversed(self.funcs.items()):
            logger.info(
                "Does one_shot_job key func : {} with function {}".format(
                    key, func
                )
            )
            func()

    def get_async_works(self):
        for w in self.topology.works:
            if w.kind is None or "one shot" not in w.kind:
                self.works.append(w)

    def has_to_stop(self):
        return not any([len(q.queue) != 0 for q in self.topology.queues])

    async def start_async_works(self):
        async with trio.open_nursery() as nursery:
            cond = trio.Condition()
            for key, af in reversed(self.async_funcs.items()):
                nursery.start_soon(af, cond)
        logger.info("Work all done in {}".format(time.time() - self.t_start))

    def start_server(self):
        t = ThreadedServer(
            MyClass,
            port=18813,
            protocol_config={"allow_public_attrs": True, "allow_pickle": True},
        )
        t.start()
        t = ThreadedServer(
            MyClass,
            port=18814,
            protocol_config={"allow_public_attrs": True, "allow_pickle": True},
        )
        t.start()
        t = ThreadedServer(
            MyClass,
            port=18815,
            protocol_config={"allow_public_attrs": True, "allow_pickle": True},
        )
        t.start()

        self.server = rpyc.connect(
            "localhost",
            18813,
            config={"allow_public_attrs": True, "allow_pickle": True},
        )
        self.server = rpyc.connect(
            "localhost",
            18814,
            config={"allow_public_attrs": True, "allow_pickle": True},
        )
        self.server = rpyc.connect(
            "localhost",
            18815,
            config={"allow_public_attrs": True, "allow_pickle": True},
        )


class multiproc:
    def __init__(self):
        pass


class Tracer(trio.abc.Instrument):
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
