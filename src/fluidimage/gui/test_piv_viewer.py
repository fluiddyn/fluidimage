import sys

from fluidimage import get_path_image_samples
from fluidimage.gui.piv_viewer import main


def test_main(monkeypatch):

    path_image_samples = get_path_image_samples()

    path = path_image_samples / "Milestone"
    monkeypatch.chdir(path)

    command = "fluidpivviewer *.h5"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()
