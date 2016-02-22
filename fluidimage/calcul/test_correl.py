
import unittest

import numpy as np

from fluidimage.synthetic import make_synthetic_images

from fluidimage.calcul.correl import (
    CorrelScipySignal, CorrelScipyNdimage, CorrelFFTNumpy, CorrelFFTW)

classes = {'sig': CorrelScipySignal, 'ndimage': CorrelScipyNdimage,
           'np_fft': CorrelFFTNumpy, 'fftw': CorrelFFTW}


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 32
        ny = 32
        displacement_x = 2.
        displacement_y = 2.

        cls.displacements = np.array([displacement_y, displacement_x])

        nb_particles = (nx // 3)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)


for k, cls in classes.items():
    def test(self):
        correl = cls(self.im0.shape, self.im1.shape)
        c = correl(self.im0, self.im1)

        inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
        self.assertTrue(np.allclose(
            self.displacements.astype('int'),
            correl.compute_displacement_from_indices(inds_max),
            rtol=1e-05, atol=1e-08))

    exec('TestCorrel.test_' + k + ' = test')

if __name__ == '__main__':
    unittest.main()
