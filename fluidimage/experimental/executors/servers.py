"""Servers
==========


"""

from multiprocessing import Process, Pipe

from threading import Thread

import trio


def launch_server(topology, type_server="multiprocessing"):

    parent_conn, child_conn = Pipe()

    # for testing
    if type_server == "multiprocessing":
        Process_ = Process
    elif type_server == "threading":
        Process_ = Thread
    else:
        raise ValueError

    process = Process_(
        target=WorkerServerMultiprocessing,
        args=(child_conn, type(topology), topology.params),
    )
    if type_server == "threading":
        process.daemon = True

    process.start()
    worker = WorkerMultiprocessing(parent_conn, process)

    return worker


class Worker:
    def __init__(self, conn, process):
        self.conn = conn
        self.process = process
        self.nb_items_to_process = 0
        self.is_available = True
        self.is_unoccupied = True

    def well_done_thanks(self):
        self.nb_items_to_process -= 1
        if self.nb_items_to_process == 0:
            self.is_unoccupied = True

    def terminate(self):
        raise NotImplementedError


class WorkerMultiprocessing(Worker):
    def send_job(self, obj):
        self.is_unoccupied = False
        self.nb_items_to_process += 1
        self.conn.send(obj)

    def new_pipe(self):
        return Pipe()

    def terminate(self):
        self.conn.send("__terminate__")
        if hasattr(self.process, "terminate"):
            self.process.terminate()


class WorkerServer:
    def __init__(self, sleep_time=0.1):
        self.sleep_time = sleep_time
        self.to_be_processed = []
        self.to_be_resent = []
        self._has_to_continue = True

        trio.run(self._start_async)

    async def _start_async(self):
        async with trio.open_nursery() as self.nursery:
            self.nursery.start_soon(self.receive)
            self.nursery.start_soon(self.launch_works)
            self.nursery.start_soon(self.send)

    async def receive(self):
        raise NotImplementedError

    async def send(self):
        raise NotImplementedError

    async def launch_works(self):
        raise NotImplementedError


class WorkerServerMultiprocessing(WorkerServer):
    def __init__(self, conn, topology_cls, params, sleep_time=0.1):
        self.conn = conn
        self.topology = topology_cls(params)
        super().__init__(sleep_time=sleep_time)

    async def receive(self):
        while self._has_to_continue:
            ret = await trio.run_sync_in_worker_thread(self.conn.recv)
            if ret == "__terminate__":
                self._has_to_continue = False
                break
            self.to_be_processed.append(ret)

    async def send(self):
        while self._has_to_continue:
            while not self.to_be_processed:
                await trio.sleep(self.sleep_time)
            work_name, key, obj, child_conn = self.to_be_processed.pop(0)
            work = self.topology.works_dict[work_name]

            def do_the_job(work, obj):
                return work.func_or_cls(obj)

            await trio.run_sync_in_worker_thread(
                child_conn.send, "computation started"
            )
            result = await trio.run_sync_in_worker_thread(do_the_job, work, obj)
            self.to_be_resent.append((work_name, key, result, child_conn))

    async def launch_works(self):
        while self._has_to_continue:
            while not self.to_be_resent:
                await trio.sleep(self.sleep_time)
            work_name, key, result, child_conn = self.to_be_resent.pop(0)

            await trio.run_sync_in_worker_thread(
                child_conn.send, (work_name, key, result)
            )
