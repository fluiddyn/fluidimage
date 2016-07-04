
from copy import deepcopy, copy

from fluidimage import logger
from .base import WaitingQueueLoadFile, WaitingQueueBase
from ...data_objects.pre_proc import ArraySerie
from ...works import load_image


class WaitingQueueLoadImageSeries(WaitingQueueLoadFile):

    def __init__(self, *args, **kwargs):
        super(WaitingQueueLoadImageSeries, self).__init__(
            'load image series', load_image, *args, **kwargs)

    def is_destination_full(self):
        cond_instance = isinstance(self.destination, WaitingQueueBase)
        buffer_limit = max(self.topology.nb_items_lim,
                           self.topology.nb_items_per_serie)

        cond_nb_items = len(self.destination) >= buffer_limit
        return (cond_instance and cond_nb_items)


class WaitingQueueMakeSerie(WaitingQueueBase):
    """
    The difference from `WaitingQueueMakeCouple` is that
    the following attributes are replaced:
    .. `self.couples` --> `self.serie_set`
    .. `self.nb_couples_to_create` --> `self.nb_serie_to_create`

    Allowing the dictionary to contain a serie of images,
    and not just a couple (2 images).

    """
    def __init__(self, name, destination,
                 work_name='make serie', topology=None):

        self.nb_series = 0
        self.ind_series = []
        self.nb_serie_to_create = {}
        self.serie_set = set()
        self.series = {}
        self.topology = topology
        work = 'make_serie'
        super(WaitingQueueMakeSerie, self).__init__(
            name, work, destination, work_name, topology)

    def is_empty(self):
        return len(self.serie_set) == 0

    def add_series(self, series):

        self.series.update({serie.get_name_files(): deepcopy(serie)
                            for serie in series})

        serie_set = [serie.get_name_files() for serie in series]
        self.serie_set.update(serie_set)
        self.nb_series = len(serie_set)
        self.ind_series = range(self.nb_series)
        nb = self.nb_serie_to_create

        for names in serie_set:
            for name in names:
                if name in nb:
                    nb[name] = nb[name] + 1
                else:
                    nb[name] = 1

    def check_and_act(self, sequential=None):
        if self.is_destination_full():
            return

        for names in copy(self.serie_set):
            if all([name in self for name in names]):
                logger.info('launch work with ' + self.work_name + repr(names))
                k0 = names[0]
                k1 = names[-1]
                newk = k0 + '-' + k1

                self.serie_set.remove(names)
                serie = self.series.pop(names)
                ind_serie = self.ind_series.pop(0)

                values = []
                logger.debug('Creating a serie for ' + repr(names))
                for name in names:
                    if self.nb_serie_to_create[name] == 1:
                        values.append(self.pop(name))
                        del self.nb_serie_to_create[name]
                        self._keys.remove(name)
                    else:
                        values.append(self[name])
                        self.nb_serie_to_create[name] -= 1

                self.destination[newk] = ArraySerie(
                    names, values, serie, ind_serie, self.nb_series)
