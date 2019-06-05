import unittest

from fluidimage import SerieOfArraysFromFiles, path_image_samples

from .image2image import Im2ImExample, apply_im2im_filter

path_src = path_image_samples / "Karman/Images"


class TestTopoExample(unittest.TestCase):
    def test_apply_im2im_filter(self):

        serie = SerieOfArraysFromFiles(path_src, "2:")
        result = apply_im2im_filter(serie)
        assert len(result) == 3
        result = apply_im2im_filter(serie, Im2ImExample, args_init=(1, 2))
        assert len(result) == 3
