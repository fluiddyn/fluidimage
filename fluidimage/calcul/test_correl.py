
from __future__ import print_function

import unittest
import logging

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes

from fluidimage import config_logging

# config_logging('debug')
logger = logging.getLogger('fluidimage')

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

        # Sometimes errors with nx = ny = 16
        # nx = 16
        # ny = 16

        #displacement_x = 2.3
        #displacement_y = 1.8
        displacement_x = 3.3
        displacement_y = 5.8

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (nx // 4)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)

for k, cls in classes.items():
    def test(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian')
        displacement_computed = np.array([dx, dy])

        logger.debug(
            k + ', displacement = [0, 0]\t error= {}\n'.format(
                abs(displacement_computed)))

        self.assertTrue(np.allclose(
            [0, 0],
            displacement_computed, atol=1e-03))

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian')

        displacement_computed = np.array([dx, dy])

        logger.debug(
            k + ', displacement = {}\t error= {}\n'.format(
                self.displacements,
                abs(displacement_computed-self.displacements)))
        
        self.assertTrue(np.allclose(
            self.displacements,
            displacement_computed,
            atol=0.8))

    exec('TestCorrel.test_correl_square_image_' + k + ' = test')

class TestCorrel1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 32
        ny = 64

        # Sometimes errors with nx = ny = 16
        # nx = 16
        # ny = 16

        displacement_x = 3.3
        displacement_y = 5.8

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (max(nx, ny) // 4)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)

for k, cls in classes.items():
    def test(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian')
        displacement_computed = np.array([dx, dy])

        logger.debug(
            k + ', displacement = [0, 0]\t error= {}\n'.format(
                abs(displacement_computed)))

        self.assertTrue(np.allclose(
            [0, 0],
            displacement_computed, atol=1e-03))

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max = correl.compute_displacement_from_correl(
            c, coef_norm=norm,
            method_subpix='2d_gaussian')

        displacement_computed = np.array([dx, dy])

        logger.debug(
            k + ', displacement = {}\t error= {}\n'.format(
                self.displacements,
                abs(displacement_computed-self.displacements)))

        self.assertTrue(np.allclose(
            self.displacements,
            displacement_computed,
            atol=0.8))

    exec('TestCorrel1.test_correl_rectangular_image_' + k + ' = test')



if __name__ == '__main__':
    unittest.main()
