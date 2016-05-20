
import unittest

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

import os
here = os.path.abspath(os.path.dirname(__file__))


class TestPIV(unittest.TestCase):

    def test_minimal_piv(self):

        params = WorkPIV.create_default_params()

        # for a very short computation
        params.piv0.shape_crop_im0 = 32
        params.piv0.grid.overlap = -2

        params.multipass.number = 2

        piv = WorkPIV(params=params)

        series = SeriesOfArrays(
            os.path.join(here, '../../../image_samples/Oseen/Images'),
            'i+1:i+3')
        serie = series.get_serie_from_index(0)

        r = piv.calcul(serie)

if __name__ == '__main__':
    unittest.main()
