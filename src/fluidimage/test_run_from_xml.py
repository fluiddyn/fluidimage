import os
import sys
import unittest

from fluidimage import get_path_image_samples
from fluidimage.run_from_xml import main

path_image_samples = get_path_image_samples()

on_linux = sys.platform == "linux"


class TestRunFromXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current_dir = os.getcwd()
        os.chdir(path_image_samples)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.current_dir)

    @unittest.skipIf(not on_linux, "Only supported on Linux")
    def test_main(self):
        path = path_image_samples / "Karman/Images.civ/0_XML/Karman_1-4.xml"
        command = f"run {str(path)} --mode recompute"
        sys.argv = command.split()

        main()
