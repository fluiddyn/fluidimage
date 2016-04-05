
from __future__ import print_function

import unittest

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes


classes = {k.replace('.', '_'): v for k, v in correlation_classes.items()}


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 64
        ny = 32
        displacement_x = 0.5
        displacement_y = 1.5

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (nx // 4)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)


for k, cls in classes.items():
    def test(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
        displacement_computed = correl.compute_displacement_from_indices(
            inds_max)

#        self.assertTrue(np.allclose(
 #           [0, 0],
  #          displacement_computed))
        print('\n', k, self.displacements, displacement_computed)

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)

        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            #method_subpix='2d_gaussian'
            method_subpix='centroid'
        )

        displacement_computed = np.array([dx, dy])

        print('\n', k, self.displacements, displacement_computed)

        #self.assertTrue(np.allclose(
        #    self.displacements,
        #    displacement_computed,
        #    atol=0.5))

    exec('TestCorrel.test_correl_' + k + ' = test')


if __name__ == '__main__':
    unittest.main()
