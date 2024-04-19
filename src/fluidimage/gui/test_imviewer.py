import sys

from fluidimage import get_path_image_samples
from fluidimage.gui.imviewer import ImageViewer, main, parse_args

path_image_samples = get_path_image_samples()


def test_fluidimviewer_version(monkeypatch):
    command = "fluidimviewer --version"
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()


def test_main(monkeypatch):

    words = ["fluidimviewer", str(path_image_samples / "Karman/Images")]
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", words)
        args = parse_args()

    ImageViewer(args)

    words = ["fluidimviewer", str(path_image_samples / "Karman/Images/*")]
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", words)
        args = parse_args()

    viewer = ImageViewer(args)

    viewer.set_autoclim(None)

    viewer._switch()
    viewer._switch()

    viewer._increase_ifile()
    viewer._decrease_ifile()

    viewer._submit_n("2")

    viewer._increase_ifile_n()
    viewer._decrease_ifile_n()

    viewer._change_cmin("1")
    viewer._change_cmax("2")
