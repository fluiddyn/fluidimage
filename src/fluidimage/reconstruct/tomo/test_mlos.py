import pytest

from fluidimage import get_path_image_samples
from fluidimage._opencv import error_import_cv2
from fluidimage.data_objects.tomo import ArrayTomoCV
from fluidimage.reconstruct.tomo import TomoMLOSCV

path_calib = str(get_path_image_samples() / "TomoPIV/calibration/cam0.h5")
path_particle = str(
    get_path_image_samples() / "TomoPIV/particle/cam0.pre/im00001a.tif"
)


@pytest.mark.usefixtures("close_plt_figs")
def test_mlos(tmp_path):
    """Test classes TomoMLOSCV and ArrayTomoCV."""

    path_output = tmp_path

    if error_import_cv2:
        with pytest.raises(ModuleNotFoundError):
            TomoMLOSCV(
                path_calib,
                xlims=(-10, 10),
                ylims=(-10, 10),
                zlims=(-5, 5),
                nb_voxels=(11, 11, 5),
            )
        return

    tomo = TomoMLOSCV(
        path_calib,
        xlims=(-10, 10),
        ylims=(-10, 10),
        zlims=(-5, 5),
        nb_voxels=(11, 11, 5),
    )
    tomo.verify_projection()
    pix = tomo.phys2pix("cam0")
    tomo.array.init_paths(path_particle, path_output)
    tomo.reconstruct(pix, path_particle, threshold=None, save=True)

    path_result = list(path_output.glob("*"))[0]
    array = ArrayTomoCV(h5file_path=path_result)
    array.describe()
    array.load_dataset(copy=True)
    array.plot_slices(0, 1)
    array.clear()
