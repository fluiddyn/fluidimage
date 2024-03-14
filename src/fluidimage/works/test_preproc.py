import pytest

from fluidimage._opencv import error_import_cv2
from fluidimage.works.preproc import Work


def _test_karman(path, backend):
    """Test preproc subpackage on image sample Karman with one index."""
    params = Work.create_default_params(backend=backend)
    params.series.path = path

    for tool in params.tools.available_tools:
        if "sliding" not in tool and "temporal" not in tool:
            tool = params.tools.__getitem__(tool)
            tool.enable = True

    if error_import_cv2 and backend == "opencv":
        with pytest.raises(ModuleNotFoundError):
            Work(params)
        return

    preproc = Work(params)
    preproc.display(1, hist=True)


@pytest.mark.usefixtures("close_plt_figs")
def test_preproc_python_karman(tmp_path_karman_small):
    _test_karman(tmp_path_karman_small, "python")


@pytest.mark.usefixtures("close_plt_figs")
def test_preproc_opencv_karman(tmp_path_karman_small):
    _test_karman(tmp_path_karman_small, "opencv")


def _test_jet(path, backend):
    """Test preproc subpackage on image sample Jet with two indices."""
    params = Work.create_default_params(backend=backend)

    params.series.path = path

    for tool in params.tools.available_tools:
        if "sliding" in tool:
            tool = params.tools.__getitem__(tool)
            tool.enable = True

    if error_import_cv2 and backend == "opencv":
        with pytest.raises(ModuleNotFoundError):
            Work(params)
        return

    preproc = Work(params)
    preproc.display(hist=False)


@pytest.mark.usefixtures("close_plt_figs")
def test_preproc_python_jet(tmp_path_jet_small):
    _test_jet(tmp_path_jet_small, "python")


@pytest.mark.usefixtures("close_plt_figs")
def test_preproc_opencv_jet(tmp_path_jet_small):
    _test_jet(tmp_path_jet_small, "opencv")
