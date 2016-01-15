
import multiprocessing
import threading
import Queue


class DataObject(object):
    def __init__(self, string):
        self.string = 'my string: ' + string

    def save(self):
        print(self.string)


class Task(object):
    def __init__(self, params=None):
        self.params = params

    def run_work_int2str(self, i):
        return DataObject(str(i))

    def run_work_clean(self, input_obj):
        input_obj.string = input_obj.string[11:]
        return input_obj


def myfunc(obj):
    obj.string = ' '.join([obj.string + '!'*3] * 3)
    return obj


# t = Task()

# o = t.run_work_clean(t.run_work_int2str(2))
# o.save()


class WaitingQueue(list):
    def __init__(self, func_work, destination):
        self.func_work = func_work
        self.destination = destination

    def is_empty(self):
        return not bool(self)

    def launch(self):
        o = self.pop()
        result = self.func_work(o)

    def fill_destination(self, result):
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
        print(self._Queue)
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
                print(result)
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


results = []
t = Task()

WaitingQueue = WaitingQueueMultiprocessing
# WaitingQueue = WaitingQueueThreading

w2 = WaitingQueue(myfunc, results)
w1 = WaitingQueue(t.run_work_clean, w2)
w0 = WaitingQueue(t.run_work_int2str, w1)
w0.extend([0, 1, 2])

queues = [w0, w1, w2]
working_works = []
while any([not q.is_empty() for q in queues]) or len(working_works) > 0:
    for q in queues:
        if not q.is_empty():
            print(q.func_work)
            working_works.append(q.launch())

    working_works[:] = [w for w in working_works if not w.fill_destination()]

for r in results:
    r.save()


# class Flux(object):
#     def __init__(self):
#         pass
