from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage import get_path_image_samples
from fluidimage.data_objects.piv import get_name_piv


def test_get_name_piv():
    path_image_samples = get_path_image_samples()
    path_jet = path_image_samples / "Jet/Images"
    series = SeriesOfArrays(path_jet)
    serie = series.get_next_serie()
    name = get_name_piv(serie)
    assert name == "piv_060a-b.h5", name
