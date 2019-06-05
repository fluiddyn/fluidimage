import pytest

from fluidimage import path_image_samples

try:
    import pyqtgraph
    from fluidimage.gui.pg_main import main, parse_args

    use_pyqtgraph = True
except ImportError:
    use_pyqtgraph = False


# @pytest.mark.skipif(not use_pyqtgraph, reason="ImportError pyqtgraph")
# def test_main0(qtbot):
#     args = []
#     args.append(str(path_image_samples / "Jet" / "Images" / "c060a.png"))
#     args = parse_args(args)
#     pg = main(args, for_testing=True)
#     qtbot.addWidget(pg)


# @pytest.mark.skipif(not use_pyqtgraph, reason="ImportError pyqtgraph")
# def test_main(qtbot):
#     args = ["--slideshow"]
#     args.extend(
#         [
#             str(path)
#             for path in (path_image_samples / "Jet" / "Images").glob(
#                 "*a.png"
#             )
#         ]
#     )
#     args = parse_args(args)
#     pg = main(args, for_testing=True)
#     qtbot.addWidget(pg)
