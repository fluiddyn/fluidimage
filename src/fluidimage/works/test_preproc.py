import pytest

from fluiddyn.io.image import imread, imsave
from fluidimage import get_path_image_samples
from fluidimage._opencv import error_import_cv2
from fluidimage.works.preproc import Work


def create_tmp_dir(tmp_path_factory, dataset: str):
    path_in = get_path_image_samples() / dataset / "Images"
    tmp_path = tmp_path_factory.mktemp("dir_" + dataset)
    for path in sorted(path_in.glob("*")):
        name = path.name
        im = imread(path)
        im = im[::6, ::6]
        imsave(tmp_path / name, im, as_int=True)
    return tmp_path


@pytest.fixture(scope="session")
def path_dir_karman(tmp_path_factory):
    return create_tmp_dir(tmp_path_factory, "Karman")


@pytest.fixture(scope="session")
def path_dir_jet(tmp_path_factory):
    return create_tmp_dir(tmp_path_factory, "Jet")


def _test_karman(path, backend):
    """Test preproc subpackage on image sample Karman with one index."""
    params = Work.create_default_params(backend=backend)
    params.preproc.series.path = path

    for tool in params.preproc.tools.available_tools:
        if "sliding" not in tool and "temporal" not in tool:
            tool = params.preproc.tools.__getitem__(tool)
            tool.enable = True

    if error_import_cv2 and backend == "opencv":
        with pytest.raises(ModuleNotFoundError):
            Work(params)
        return

    preproc = Work(params)
    preproc.display(1, hist=True)


def test_preproc_python_karman(path_dir_karman):
    _test_karman(path_dir_karman, "python")


def test_preproc_opencv_karman(path_dir_karman):
    _test_karman(path_dir_karman, "opencv")


def _test_jet(path, backend):
    """Test preproc subpackage on image sample Jet with two indices."""
    params = Work.create_default_params(backend=backend)

    params.preproc.series.path = path

    for tool in params.preproc.tools.available_tools:
        if "sliding" in tool:
            tool = params.preproc.tools.__getitem__(tool)
            tool.enable = True

    if error_import_cv2 and backend == "opencv":
        with pytest.raises(ModuleNotFoundError):
            Work(params)
        return

    preproc = Work(params)
    preproc.display(hist=False)


def test_preproc_python_jet(path_dir_jet):
    _test_jet(path_dir_jet, "python")


def test_preproc_opencv_jet(path_dir_jet):
    _test_jet(path_dir_jet, "opencv")
