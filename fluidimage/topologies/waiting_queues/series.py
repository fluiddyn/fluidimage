"""
Waiting queues for series (:mod:`fluidimage.topologies.waiting_queues.series`)
==============================================================================

.. autoclass:: WaitingQueueLoadImageSeries
   :members:
   :private-members:

.. autoclass:: WaitingQueueMakeSerie
   :members:
   :private-members:

"""


from __future__ import print_function
import os
from copy import deepcopy, copy

from ...util.util import logger
from .base import (
    WaitingQueueLoadImage,
    WaitingQueueBase,
    WaitingQueueMultiprocessing,
)

from ...data_objects.preproc import ArraySerie


class WaitingQueueLoadImageSeries(WaitingQueueLoadImage):
    """Waiting queue for loading series of images."""

    def __init__(self, *args, **kwargs):
        self.sequential = kwargs.pop("sequential")
        super(WaitingQueueLoadImageSeries, self).__init__(*args, **kwargs)

    def is_destination_full(self):
        cond_instance = isinstance(self.destination, WaitingQueueBase)
        buffer_limit = max(
            self.topology.nb_items_lim, self.topology.nb_items_per_serie
        )

        cond_nb_items = len(self.destination) >= buffer_limit

        return cond_instance and cond_nb_items

    def add_name_files(self, names):
        self.update(
            {name: os.path.join(self.path_dir, name) for name in names}, names
        )

    def check_and_act(self, *args, **kwargs):
        kwargs["sequential"] = self.sequential
        super(WaitingQueueLoadImageSeries, self).check_and_act(*args, **kwargs)


class WaitingQueueMakeSerie(WaitingQueueBase):
    """
    The difference from `WaitingQueueMakeCouple` is that
    the following attributes are replaced:

    - `self.couples` --> `self.serie_set`
    - `self.nb_couples_to_create` --> `self.nb_serie_to_create`

    Allowing the dictionary to contain a serie of images,
    and not just a couple (2 images).

    """

    def __init__(self, name, destination, work_name="make serie", topology=None):

        self.nb_series = 0
        self.ind_series = []
        self.nb_serie_to_create = {}
        self.serie_set = set()
        self.series = {}
        self.topology = topology
        work = "make_serie"
        super(WaitingQueueMakeSerie, self).__init__(
            name, work, destination, work_name, topology
        )

    def is_empty(self):
        return len(self.serie_set) == 0

    def add_series(self, series):

        self.series.update(
            {serie.get_name_arrays(): deepcopy(serie) for serie in series}
        )

        serie_set = [serie.get_name_arrays() for serie in series]
        self.serie_set.update(serie_set)
        self.nb_series = len(serie_set)
        self.ind_series = list(range(self.nb_series))
        nb = self.nb_serie_to_create

        for names in serie_set:
            for name in names:
                if name in nb:
                    nb[name] = nb[name] + 1
                else:
                    nb[name] = 1

    def is_destination_full(self):

        cond_instance = isinstance(self.destination, WaitingQueueMultiprocessing)

        cond_nb_items = len(self.destination) >= self.topology.nb_items_lim

        return cond_instance and cond_nb_items

    def check_and_act(self, sequential=None):

        for names in copy(self.serie_set):
            if self.is_destination_full():
                break

            if all([name in self for name in names]):
                k0 = names[0]
                k1 = names[-1]
                newk = k0 + "-" + k1
                logger.info("launch work %s with %s", self.work_name, newk)

                self.serie_set.remove(names)
                serie = self.series.pop(names)
                ind_serie = self.ind_series.pop(0)

                values = []
                logger.debug("Creating a serie for " + repr(names))
                for name in names:
                    if self.nb_serie_to_create[name] == 1:
                        values.append(self.pop(name))
                        del self.nb_serie_to_create[name]
                        self._keys.remove(name)
                    else:
                        values.append(self[name])
                        self.nb_serie_to_create[name] -= 1

                self.destination[newk] = ArraySerie(
                    names, values, serie, ind_serie, self.nb_series
                )
