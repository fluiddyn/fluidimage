
import multiprocessing
import threading
import Queue

from data_objects import ArrayCouple


class WaitingQueueBase(dict):
    def __init__(self, name, work, destination=None, work_name=None):
        self.name = name
        self.work = work
        self.destination = destination
        self.work_name = work_name

    def is_empty(self):
        return not bool(self)

    def check_and_act(self):
        k, o = self.popitem()
        result = self.work(o)
        self.fill_destination(k, result)

    def fill_destination(self, k, result):
        if self.destination is not None:
            self.destination[k] = result


class WaitingQueueMultiprocessing(WaitingQueueBase):

    @staticmethod
    def _Queue(*args, **kwargs):
        return multiprocessing.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return multiprocessing.Process(*args, **kwargs)

    def check_and_act(self):
        k, o = self.popitem()
        comm = self._Queue()

        def f(comm):
            result = self.work(o)
            comm.put(result)

        p = self._Process(target=f, args=(comm,))
        p.start()

        def fill_destination():
            if p.is_alive():
                return False
            else:
                result = comm.get()
                self.fill_destination(k, result)
                return True

        p.fill_destination = fill_destination
        return [p]


class WaitingQueueThreading(WaitingQueueMultiprocessing):
    @staticmethod
    def _Queue(*args, **kwargs):
        return Queue.Queue(*args, **kwargs)

    @staticmethod
    def _Process(*args, **kwargs):
        return threading.Thread(*args, **kwargs)


class WaitingQueueLoadFile(WaitingQueueThreading):
    def __init__(self, *args, **kwargs):
        self.path_dir = kwargs.pop('path_dir')
        super(WaitingQueueLoadFile, self).__init__(*args, **kwargs)
        self.work_name = 'load'

    def add_name_files(self, names):
        self.update({name: self.path_dir + '/' + name for name in names})


class WaitingQueueMakeCouple(WaitingQueueBase):
    def __init__(self, name, destination):
        self.name = name
        self.destination = destination
        self.work_name = 'make couples'

        self.nb_couples_to_create = {}
        self.work = 'make_couples'

        self.couples = set()

    def add_couples(self, couples):

        self.couples.update(couples)
        nb = self.nb_couples_to_create

        for couple in couples:
            for name in couple:
                if name in nb:
                    nb[name] = nb[name] + 1
                else:
                    nb[name] = 1

    def check_and_act(self):

        for k0 in self.keys():
            for k1 in self.keys():
                if (k1, k0) in self.couples:
                    k0, k1 = k1, k0

                if (k0, k1) in self.couples:
                    newk = k0 + '-' + k1
                    self.couples.remove((k0, k1))

                    if self.nb_couples_to_create[k0] == 1:
                        v0 = self.pop(k0)
                        del self.nb_couples_to_create[k0]
                    else:
                        v0 = self[k0]
                        self.nb_couples_to_create[k0] = \
                            self.nb_couples_to_create[k0] - 1

                    if self.nb_couples_to_create[k1] == 1:
                        v1 = self.pop(k1)
                        del self.nb_couples_to_create[k1]
                    else:
                        v1 = self[k1]
                        self.nb_couples_to_create[k1] = \
                            self.nb_couples_to_create[k1] - 1

                    self.destination[newk] = ArrayCouple((k0, k1), (v0, v1))