"""Splitters to split a topology task

Splitters are used for :class:`fluidimage.executors.multi_exec_subproc.MultiExecutorSubproc`.

.. autoclass:: Splitter
   :members:
   :private-members:

.. autoclass:: SplitterCompleteAware
   :members:
   :private-members:

.. autoclass:: SplitterFromSeries
   :members:
   :private-members:

.. autoclass:: SplitterFromImages
   :members:
   :private-members:

"""

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles, SeriesOfArrays


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


def split_list(sequence, num_parts):
    num_parts = min(num_parts, len(sequence))
    k, m = divmod(len(sequence), num_parts)
    return [
        sequence[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(num_parts)
    ]


class Splitter(ABC):

    def __init__(self, params, num_processes, topology=None):
        """Initialize the splitter"""
        self.params = params
        self.num_processes = num_processes

    @abstractmethod
    def iter_over_new_params(self):
        """Split the work in approximately equal subworks"""


class SplitterCompleteAware(Splitter):
    _path_dir_indices: Path
    indices_lists: list
    _indices_files_saved: bool
    ranges: list

    @abstractmethod
    def _get_params_things(self, params):
        """Get the Parameters object corresponding to the series or the images"""

    def save_indices_files(self, path_dir):
        self._path_dir_indices = path_dir

        for idx_process, indices in enumerate(self.indices_lists):
            if not indices:
                continue
            path = path_dir / f"indices{idx_process:03}.txt"
            path.write_text("\n".join(str(index) for index in indices) + "\n")

        self._indices_files_saved = True

    def _iter_over_new_params_from_indices_lists(self):
        if not self._indices_files_saved:
            raise RuntimeError("First call save_indices_files.")
        path_dir = self._path_dir_indices

        params0 = deepcopy(self.params)
        params0.saving.how = "from_path_indices"

        p_series = self._get_params_things(params0)
        p_series._set_attrib("path_indices_file", None)

        for idx_process, indices in enumerate(self.indices_lists):
            if not indices:
                continue
            params = deepcopy(params0)
            p_series = self._get_params_things(params)
            p_series.path_indices_file = path_dir / f"indices{idx_process:03}.txt"
            yield params

    @abstractmethod
    def _iter_over_new_params_from_ranges(self):
        """Iter from self.ranges"""

    def iter_over_new_params(self):
        if self.ranges is not None:
            yield from self._iter_over_new_params_from_ranges()
        elif self.indices_lists is not None:
            yield from self._iter_over_new_params_from_indices_lists()
        else:
            assert False


class SplitterFromSeries(SplitterCompleteAware):

    def __init__(self, params, num_processes, topology=None):
        super().__init__(params, num_processes, topology=topology)

        if topology is None:
            p_series = self.get_params_series(params)
            self.series = SeriesOfArrays(
                p_series.path,
                p_series.str_subset,
                ind_start=p_series.ind_start,
                ind_stop=p_series.ind_stop,
                ind_step=p_series.ind_step,
            )
        else:
            self.series = topology.series

        self.indices_lists = None
        self.ranges = None
        self._indices_files_saved = False
        self._path_dir_indices = None

        if (
            topology is not None
            and topology.how_saving == "complete"
            and hasattr(topology, "compute_indices_to_be_computed")
        ):
            indices = topology.compute_indices_to_be_computed()
            self.indices_lists = split_list(indices, self.num_processes)
            self.num_expected_results = len(indices)
        else:
            self.num_expected_results = len(
                range(
                    self.series.ind_start,
                    self.series.ind_stop,
                    self.series.ind_step,
                )
            )
            self.ranges = split_range(
                self.series.ind_start,
                self.series.ind_stop,
                self.series.ind_step,
                self.num_processes,
            )

    def get_params_series(self, params):
        return params.series

    def _get_params_things(self, params):
        return self.get_params_series(params)

    def _iter_over_new_params_from_ranges(self):
        for sss in self.ranges:
            if len(range(*sss)) == 0:
                continue
            params = deepcopy(self.params)
            p_series = self.get_params_series(params)
            p_series.ind_start, p_series.ind_stop, p_series.ind_step = sss
            yield params


class SplitterFromImages(SplitterCompleteAware):

    def __init__(self, params, num_processes, topology=None):
        super().__init__(params, num_processes, topology=topology)

        if topology is None:
            p_images = self.get_params_images(params)
            self.serie = SerieOfArraysFromFiles(
                p_images.path,
                p_images.str_subset,
            )
        else:
            self.serie = topology.serie

        self.indices_lists = None
        self.ranges = None
        self._indices_files_saved = False
        self._path_dir_indices = None

        if (
            topology is not None
            and topology.how_saving == "complete"
            and hasattr(topology, "compute_indices_to_be_computed")
        ):
            indices = topology.compute_indices_to_be_computed()
            self.indices_lists = split_list(indices, self.num_processes)
            self.num_expected_results = len(indices)
        else:
            self.num_expected_results = len(self.serie)
            slicing_tuples = self.serie.get_slicing_tuples()
            s0 = slicing_tuples[0]
            self.ranges = split_range(s0[0], s0[1], s0[2], self.num_processes)

            if len(slicing_tuples) == 1:
                self.slicing_str_post = ""
            else:
                self.slicing_str_post = "," + ",".join(
                    ":".join(str(n) for n in sss) for sss in slicing_tuples[1:]
                )

    def get_params_images(self, params):
        return params.images

    def _get_params_things(self, params):
        return self.get_params_images(params)

    def _iter_over_new_params_from_ranges(self):
        for sss in self.ranges:
            if len(range(*sss)) == 0:
                continue
            params = deepcopy(self.params)
            p_images = self.get_params_images(params)
            p_images.str_subset = (
                ":".join(str(n) for n in sss) + self.slicing_str_post
            )
            yield params
