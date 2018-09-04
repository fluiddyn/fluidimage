"""Servers
==========


"""

import trio



class WorkerClient:
    def __init__(self):
        pass


class WorkerClientMultiprocessing(WorkerClient):
    pass


class WorkerServer:
    def __init__(self, sleep_time=0.1):
        self.sleep_time = sleep_time
        self.to_be_processed = []
        self.to_be_resend = []
        self._has_to_continue = True

    async def _start_async(self):
        async with trio.open_nursery() as self.nursery:
            self.nursery.start_soon(self.receiver)
            self.nursery.start_soon(self.launch_works)
            self.nursery.start_soon(self.sender)

    async def receiver(self):
        pass

    async def sender(self):
        pass

    async def launch_works(self):
        pass


class WorkerServerMultiprocessing(WorkerServer):

    def __init__(self, pipe, sleep_time=0.1):
        self.pipe = pipe
        super().__init__(sleep_time=sleep_time)

    async def receiver(self):
        while self._has_to_continue:
            trio.sleep(self.sleep_time)

    async def sender(self):
        while self._has_to_continue:
            trio.sleep(self.sleep_time)

    async def launch_works(self):
        while self._has_to_continue:
            trio.sleep(self.sleep_time)
