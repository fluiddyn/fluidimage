"""Splitters to split a topology task


"""

from abc import ABC, abstractmethod
from copy import deepcopy

from fluiddyn.util.serieofarrays import SeriesOfArrays


def split_range(start0, stop0, step0, num_parts):
    num_elems = len(range(start0, stop0, step0))

    num_elems_per_parts_approx = num_elems // num_parts
    remainder = num_elems % num_parts

    assert num_elems == num_elems_per_parts_approx * num_parts + remainder

    num_elems_vs_ipart = [num_elems_per_parts_approx] * num_parts
    for ipart in range(remainder):
        num_elems_vs_ipart[ipart] += 1

    assert sum(num_elems_vs_ipart) == num_elems

    ranges = []
    start = start0
    for ipart, num_elems_ipart in enumerate(num_elems_vs_ipart):
        stop = start + num_elems_ipart * step0
        ranges.append((start, stop, step0))
        start = stop

    return ranges


class Splitter(ABC):

    def __init__(self, params, num_processes):
        """Initialize the splitter"""
        self.params = params
        self.num_processes = num_processes

    @abstractmethod
    def iter_over_new_params(self):
        """Split the work in approximately equal subworks"""


class SplitterFromSeries(Splitter):

    def __init__(self, params, num_processes):
        super().__init__(params, num_processes)

        p_series = self.get_params_series(params)
        self.series = SeriesOfArrays(
            p_series.path,
            p_series.str_subset,
            ind_start=p_series.ind_start,
            ind_stop=p_series.ind_stop,
            ind_step=p_series.ind_step,
        )
        self.ranges = split_range(
            self.series.ind_start,
            self.series.ind_stop,
            self.series.ind_step,
            self.num_processes,
        )

    def get_params_series(self, params):
        return params.series

    def iter_over_new_params(self):
        for sss in self.ranges:
            params = deepcopy(self.params)
            p_series = self.get_params_series(params)
            p_series.ind_start, p_series.ind_stop, p_series.ind_step = sss
            yield params
