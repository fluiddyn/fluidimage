
from __future__ import print_function

import unittest

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes
import pylab

classes = {k.replace('.', '_'): v for k, v in correlation_classes.items()}

try:
    from reikna.cluda import any_api
    api = any_api()
except Exception:
    classes.pop('cufft')

try:
    import pycuda
except ImportError:
    classes.pop('pycuda')

try:
    import skcuda
except ImportError:
    classes.pop('skcufft')

try:
    import theano
except ImportError:
    classes.pop('theano')


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 32
        ny = 32
        displacement_x = 3.3
        displacement_y = 5.8

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (nx // 4)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)
        #pylab.imshow(cls.im0)
        #pylab.show()

for k, cls in classes.items():
    def test(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian'
            #method_subpix='centroid'
        )
        displacement_computed = np.array([dx, dy])
#        inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
#        displacement_computed = correl.compute_displacement_from_indices(
#            inds_max)

        self.assertTrue(np.allclose(
            [0, 0],
            displacement_computed, atol=1e-05))
        print('\n', k, ', displacement = ', [0, 0],
              '\n\t error=', np.abs(displacement_computed))
        
        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian'
            #method_subpix='centroid'
        )

        displacement_computed = np.array([dx, dy])
        print()

        print(k, ', displacement = ', self.displacements,
              '\n\t error=', np.abs(displacement_computed-self.displacements), '\n')

        self.assertTrue(np.allclose(
            self.displacements,
            displacement_computed,
            atol=0.5))

    exec('TestCorrel.test_correl_' + k + ' = test')


if __name__ == '__main__':
    unittest.main()
