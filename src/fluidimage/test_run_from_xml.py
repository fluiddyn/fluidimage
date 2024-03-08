import shutil
import sys

from fluidimage import get_path_image_samples
from fluidimage.piv import TopologyPIV
from fluidimage.run_from_xml import main

path_image_samples = get_path_image_samples()


def test_main(monkeypatch):

    monkeypatch.chdir(path_image_samples)

    path = path_image_samples / "Karman/Images.civ/0_XML/Karman_1-4.xml"
    command = f"run {path} --mode recompute"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()


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
