
import unittest

from fluiddyn.io import stdout_redirected

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

from fluidimage import path_image_samples


class TestPIV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        series = SeriesOfArrays(
            str(path_image_samples / "Oseen/Images/Oseen*"), "i+1:i+3"
        )
        cls.serie = series.get_serie_from_index(0)

    def test_minimal_piv(self):

        params = WorkPIV.create_default_params()

        # for a very short computation
        params.piv0.shape_crop_im0 = 32
        params.piv0.grid.overlap = -3

        params.multipass.number = 2

        piv = WorkPIV(params=params)

        with stdout_redirected():
            piv.calcul(self.serie)

    def test_piv_list(self):

        params = WorkPIV.create_default_params()

        # for a very short computation
        params.piv0.shape_crop_im0 = [33, 44]
        params.piv0.grid.overlap = -3

        params.multipass.use_tps = False

        piv = WorkPIV(params=params)

        with stdout_redirected():
            piv.calcul(self.serie)


if __name__ == "__main__":
    unittest.main()
