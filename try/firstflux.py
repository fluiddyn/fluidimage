
from time import time, sleep

import multiprocessing
import threading
import Queue


class DataObject(object):
    def __init__(self, string):
        self.s = 'my string: ' + string

    def save(self):
        print('save: simulate long io, sleep 1 s')
        sleep(1)
        print(self.s)


class Task(object):
    def __init__(self, params=None):
        self.params = params

    def run_work0(self, i):
        print('work0: simulate long io, sleep 1 s')
        sleep(1)
        return DataObject(str(i))

    def run_work1(self, obj):
        obj.s = obj.s[11:]
        return obj


def run_work2(obj):
    obj.s = ' '.join([obj.s + '!'*3] * 3)
    return obj


task = Task()

t = time()
o = run_work2(task.run_work1(task.run_work0(2)))
o.save()
print('time one process serial: {} s'.format(time() - t))


class WaitingQueue(list):
    def __init__(self, func_work, destination=None):
        self.func_work = func_work
        self.destination = destination

    def is_empty(self):
        return not bool(self)

    def launch(self):
        o = self.pop()
        result = self.func_work(o)

    def fill_destination(self, result):
        if self.destination is not None:
            self.destination.append(result)


class WaitingQueueMultiprocessing(WaitingQueue):

    @staticmethod
    def _Queue(*args, **kwargs):
        return multiprocessing.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return multiprocessing.Process(*args, **kwargs)

    def launch(self):
        o = self.pop()
        comm = self._Queue()

        def f(comm):
            result = self.func_work(o)
            comm.put(result)

        p = self._Process(target=f, args=(comm,))
        p.start()

        def fill_destination():
            if p.is_alive():
                return False
            else:
                result = comm.get()
                self.fill_destination(result)
                return True

        p.fill_destination = fill_destination
        return p


class WaitingQueueThreading(WaitingQueueMultiprocessing):
    @staticmethod
    def _Queue(*args, **kwargs):
        return Queue.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return threading.Thread(*args, **kwargs)


WaitingQueue = WaitingQueueMultiprocessing
# WaitingQueue = WaitingQueueThreading

w3 = WaitingQueue(lambda o: o.save())
w2 = WaitingQueue(run_work2, w3)
w1 = WaitingQueue(task.run_work1, w2)
w0 = WaitingQueue(task.run_work0, w1)
w0.extend(range(5))


def run_flux(queues):
    working_works = []
    while any([not q.is_empty() for q in queues]) or len(working_works) > 0:
        for q in queues:
            if not q.is_empty():
                print(q.func_work)
                working_works.append(q.launch())

        working_works[:] = [w for w in working_works
                            if not w.fill_destination()]


queues = [w0, w1, w2, w3]

t = time()
run_flux(queues)
print('time five processes parallel: {} s'.format(time() - t))
