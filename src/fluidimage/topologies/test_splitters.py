from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage import get_path_image_samples
from fluidimage.topologies.piv import TopologyPIV
from fluidimage.topologies.splitters import (
    SplitterFromSeries,
    split_list,
    split_range,
)

path_image_samples = get_path_image_samples()


def test_split_range():
    for start, stop, step, num_parts in ((1, 55, 2, 9), (0, 10, 1, 3)):
        num_elems = len(range(start, stop, step))
        ranges = split_range(start, stop, step, num_parts)
        indices = []
        for _r in ranges:
            indices.extend(range(*_r))
        assert len(set(indices)) == num_elems
        assert sum(len(range(*sss)) for sss in ranges) == num_elems


def test_split_list():
    num_processes = 4
    for seq in ([5, 1, 3], list(range(10))):
        lists = split_list(seq, num_processes)
        assert len(lists) == min([num_processes, len(seq)])
        indices = []
        for _l in lists:
            indices.extend(_l)
        assert indices == seq


def test_splitter_from_serie():

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_image_samples / "Jet/Images")

    params.saving.how = "recompute"
    params.saving.postfix = "test_piv_splitter"

    splitter = SplitterFromSeries(params, 4)

    p_series = splitter.get_params_series(params)

    series = SeriesOfArrays(
        p_series.path,
        p_series.str_subset,
        ind_start=p_series.ind_start,
        ind_stop=p_series.ind_stop,
        ind_step=p_series.ind_step,
    )
    indices = list(range(series.ind_start, series.ind_stop, series.ind_step))

    indices_new = []
    for params_split in splitter.iter_over_new_params():
        p_series = splitter.get_params_series(params_split)
        indices_new.extend(
            list(range(p_series.ind_start, p_series.ind_stop, p_series.ind_step))
        )

    assert indices == indices_new
