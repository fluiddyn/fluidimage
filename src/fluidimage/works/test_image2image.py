import pytest

from fluidimage import get_path_image_samples
from fluidimage.works.image2image import Work


@pytest.mark.usefixtures("close_plt_figs")
def test_work_image2image():

    params = Work.create_default_params()

    # for a class (with one argument for the function init):
    params.im2im = "fluidimage.image2image.Im2ImExample"
    params.args_init = ((1024, 2048), "clip")

    params.images.path = get_path_image_samples() / "Jet/Images/c*"
    params.images.str_subset = "60:,:"

    work = Work(params)

    work.display()
