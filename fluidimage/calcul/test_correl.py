
import unittest
from copy import deepcopy

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes


classes = {k.replace('.', '_'): v for k, v in correlation_classes.items()}


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 16
        ny = 16
        displacement_x = 1.
        displacement_y = 1.

        cls.displacements = np.array([displacement_y, displacement_x])

        nb_particles = (nx // 3)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)


for k, cls in classes.items():
    def test(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)
        c = correl(self.im0, self.im1)

        inds_max = np.array(np.unravel_index(c.argmax(), c.shape))

        displacement_computed = correl.compute_displacement_from_indices(
            inds_max)

        self.assertTrue(np.allclose(
            self.displacements.astype('int'),
            displacement_computed))

    exec('TestCorrel.test_' + k + ' = test')


if __name__ == '__main__':
    unittest.main()
