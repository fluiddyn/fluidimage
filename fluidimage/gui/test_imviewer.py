import sys
import unittest

from fluiddyn.io import stdout_redirected
from fluidimage import path_image_samples
from fluidimage.gui.imviewer import ImageViewer, parse_args


class TestImageViewer(unittest.TestCase):
    def test_main(self):
        with stdout_redirected():
            command = "fluidimviewer"
            args = command.split()
            args.append(str(path_image_samples / "Karman/Images"))
            sys.argv = args
            args = parse_args()
            ImageViewer(args)

            args = command.split()
            args.append(str(path_image_samples / "Karman/Images/*"))
            sys.argv = args

            args = parse_args()
            self = ImageViewer(args)

            self.set_autoclim(None)

            self._switch()
            self._switch()

            self._increase_ifile()
            self._decrease_ifile()

            self._submit_n("2")

            self._increase_ifile_n()
            self._decrease_ifile_n()

            self._change_cmin("1")
            self._change_cmax("2")
