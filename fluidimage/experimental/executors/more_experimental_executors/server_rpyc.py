import os
import multiprocessing
import rpyc
from rpyc.utils.server import ThreadedServer


######


class MyClass(rpyc.Service):
    """ Manages jobs requested on a Node instance"""

    def on_connect(self):
        # code that runs when a connection is created
        # (to init the serivce, if needed)
        self.result_queue = multiprocessing.Queue
        self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        self.results = []
        print("connected")
        pass

    def on_disconnect(self):
        # code that runs when the connection has already closed
        # (to finalize the service, if needed)
        pass

    def function(self, work, obj):
        print("start working a job")
        self.results.append(work(obj))
        print("append new result")

    def exposed_add_work(self, work, obj):
        """Adds a new job to the node job execution queue and executes it"""
        # execute the job in a new thread
        print("adding a job")

        self.function(work, obj)
        # myJobWorker = multiprocessing.Process(target=self.function, args=(work, obj,),)
        # myJobWorker.start()
        # myJobWorker.join()

    def exposed_get_a_result(self):
        if self.results:
            return self.results.pop(0)
        else:
            return None

    def exposed_result_ready(self):
        return len(self.results) != 0


def startNode():
    # retrieve node name from node configuration file
    # appConfig  = util.config()
    t = ThreadedServer(
        MyClass,
        port=18813,
        protocol_config={"allow_public_attrs": True, "allow_pickle": True},
    )
    t.start()


######

if __name__ == "__main__":
    startNode()
