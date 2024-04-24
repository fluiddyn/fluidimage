import shutil
import sys

import pytest

from fluidimage import get_path_image_samples
from fluidimage.piv import TopologyPIV
from fluidimage.run_from_xml import main

path_image_samples = get_path_image_samples()


@pytest.mark.parametrize("name", ["Karman", "Jet"])
def test_uvmat(tmp_path, monkeypatch, name):

    path_dir_images = tmp_path / "Images"
    path_dir_images.mkdir()

    for path_im in (path_image_samples / name / "Images").glob("*"):
        shutil.copy(path_im, path_dir_images)

    path_save_uvmat_xml = (
        path_image_samples / name /"Images.civ/0_XML/instructions.xml"
    )
    text = path_save_uvmat_xml.read_text()
    text = text.replace("TO_BE_REPLACED_BY_TMP_PATH", str(tmp_path))
    path_uvmat_xml = tmp_path / "uvmat.xml"
    path_uvmat_xml.write_text(text)

    command = f"run {path_uvmat_xml} --mode recompute"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        action = main()

    path_results = tmp_path / "Images.civ"
    assert action.params.saving.path == str(path_results)

    paths_piv = sorted(p.name for p in path_results.glob("piv*.h5"))
    assert paths_piv == ["piv_01-02.h5", "piv_03-04.h5"]


def test_piv_sequential(tmp_path, monkeypatch):

    path_dir_images = tmp_path / "Images"
    path_dir_images.mkdir()

    for path_im in (path_image_samples / "Jet/Images").glob("*"):
        shutil.copy(path_im, path_dir_images)

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_dir_images)

    params.piv0.shape_crop_im0 = 128
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.saving.how = "recompute"
    params.saving.postfix = "test_piv_run_from_xml"

    params._set_child(
        "compute_kwargs",
        attribs={"executor": "exec_async_sequential", "nb_max_workers": 1},
    )

    path_params = path_dir_images.parent / "params.xml"

    params._save_as_xml(path_file=path_params)
    assert path_params.exists()

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", ["run", str(path_params)])
        main()
