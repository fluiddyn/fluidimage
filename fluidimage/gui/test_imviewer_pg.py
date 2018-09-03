
import unittest

# import sys
# from pyqtgraph.Qt import QTest

try:
    import pyqtgraph
    from fluidimage.gui.pg_main import parse_args, main

    use_pyqtgraph = True
except ImportError:
    use_pyqtgraph = False

from fluiddyn.io import stdout_redirected

from fluidimage import path_image_samples


@unittest.skip(
    "PyQtGraph is not installed. Test disabled because it is not " "automated"
)
class TestImageViewerPG(unittest.TestCase):
    def test_main(self):
        with stdout_redirected():
            args = []
            args.append(str(path_image_samples / "Jet" / "Images" / "c060a.png"))
            args = parse_args(args)
            pg = main(args)
            # pg.app.exit(0)

            args = ["--slideshow"]
            args.extend(
                [
                    str(path)
                    for path in (path_image_samples / "Jet" / "Images").glob(
                        "*a.png"
                    )
                ]
            )
            args = parse_args(args)
            pg = main(args)
            # pg.app.exit(0)
