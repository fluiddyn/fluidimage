import pytest

from fluidimage import get_path_image_samples
from fluidimage.data_objects.display_piv import DisplayPIV
from fluidimage.data_objects.piv import LightPIVResults, MultipassPIVResults
from fluidimage.works.piv import WorkPIV

path_images = get_path_image_samples() / "Oseen/Images"


class MyObj:
    pass


@pytest.mark.usefixtures("close_plt_figs")
def test_minimal_piv(tmp_path):
    params = WorkPIV.create_default_params()

    # for a very short computation
    params.piv0.shape_crop_im0 = 32
    params.piv0.grid.overlap = -3

    # still buggy
    # params.piv0.nb_peaks_to_search = 2

    params.multipass.number = 2

    params.fix.displacement_max = 2
    params.fix.threshold_diff_neighbour = 2

    params.series.path = str(path_images / "Oseen*")
    params.series.str_subset = "i+1:i+3"

    piv = WorkPIV(params=params)

    result = piv.process_1_serie()

    result.piv0.save(tmp_path)
    result.save(tmp_path)

    path_file = next(tmp_path.iterdir())
    MultipassPIVResults(path_file)

    result.save(tmp_path, "uvmat")

    light = result.make_light_result()

    light.save(tmp_path)
    path_file = next(tmp_path.glob("*light*"))

    LightPIVResults(str_path=str(path_file))

    serie = piv.get_serie(0)
    arrays = serie.get_arrays()
    result = piv.calcul_from_arrays(*arrays)
    path_file = result.save(tmp_path)
    type(result)(str_path=path_file)


@pytest.mark.usefixtures("close_plt_figs")
def test_piv_list():

    params = WorkPIV.create_default_params()

    # for a very short computation
    params.piv0.shape_crop_im0 = [33, 44]
    params.piv0.grid.overlap = -3
    # params.piv0.nb_peaks_to_search = 2

    params.multipass.use_tps = False

    params.series.path = str(path_images / "Oseen*")
    params.series.str_subset = "i+1:i+3"

    piv = WorkPIV(params=params)

    result = piv.process_1_serie()

    piv0 = result.piv0
    im0, im1 = piv0.get_images()
    DisplayPIV(im0, im1, piv0)
    d = DisplayPIV(im0, im1, piv0, show_interp=True, hist=True)

    d.switch()
    d.select_arrow([0], artist=d.q)
    d.select_arrow([0], artist=None)

    event = MyObj()

    event.key = "alt+h"
    event.inaxes = None
    d.onclick(event)

    event.inaxes = d.ax1

    event.key = "alt+s"
    d.onclick(event)
    event.key = "alt+left"
    d.onclick(event)
    event.key = "alt+right"
    d.onclick(event)

    event.artist = None
    d.onpick(event)

    event.artist = d.q
    event.ind = [0]
    d.onpick(event)
