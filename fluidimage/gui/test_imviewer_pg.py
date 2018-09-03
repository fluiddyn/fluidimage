
import unittest

# import sys
# from pyqtgraph.Qt import QTest

from fluiddyn.io import stdout_redirected

from fluidimage.gui.pg_main import parse_args, main
from fluidimage import path_image_samples


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
