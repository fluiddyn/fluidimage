class ExecuterBase:
    def __init__(self, topology):
        self.topology = topology
        self.queues = []
        #assigne dict to queue
        for q in self.topology.queues:
            new_queue = {}
            self.queues.append(new_queue)
            q.queue = self.queues[-1]

