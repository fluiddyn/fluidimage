import sys

from fluidimage import get_path_image_samples
from fluidimage.gui.piv_viewer import main


def test_main(monkeypatch):

    path_image_samples = get_path_image_samples()
    path = path_image_samples / "Milestone"

    command = f"fluidpivviewer {path}"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        viewer = main()

    assert viewer.index_file == 0

    path = path / "piv_0000a-b.h5"
    command = f"fluidpivviewer {path}"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        viewer = main()

    assert viewer.index_file == 0

    viewer._increment_index()
    assert viewer.index_file == 1

    viewer._change_index_from_textbox("10")
    assert viewer.index_file == 1

    viewer._change_index_from_textbox("-1")
    assert viewer.index_file == 0

    viewer._decrement_index()

    viewer._change_index_from_textbox("abc")
