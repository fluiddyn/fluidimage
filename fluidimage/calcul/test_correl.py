
from __future__ import print_function

import unittest
import logging

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import (
    correlation_classes,
    CorrelScipySignal,
    CorrelTheano,
    CorrelPythran,
    CorrelPyCuda,
    CorrelFFTBase,
    # CorrelBase,
)

# config_logging('debug')
logger = logging.getLogger("fluidimage")

classes = {k.replace(".", "_"): v for k, v in correlation_classes.items()}
classes2 = {
    "sig": CorrelScipySignal,
    "theano": CorrelTheano,
    "pycuda": CorrelPyCuda,
    "pythran": CorrelPythran,
}

try:
    from reikna.cluda import any_api

    api = any_api()
except Exception:
    classes.pop("cufft")

try:
    import pycuda
except ImportError:
    classes.pop("pycuda")
    classes2.pop("pycuda")

try:
    import skcuda
except ImportError:
    classes.pop("skcufft")

try:
    import theano
except ImportError:
    classes.pop("theano")
    classes2.pop("theano")


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 32
        ny = 32

        # Sometimes errors with nx = ny = 16
        # nx = 16
        # ny = 16

        displacement_x = 3.3
        displacement_y = 5.8

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (nx // 4) ** 2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.
        )


for k, cls in classes.items():

    def _test(self, cls=cls, k=k):

        if issubclass(cls, CorrelFFTBase):
            displacement_max = "50%"
        else:
            displacement_max = None

        correl = cls(
            self.im0.shape, self.im1.shape, displacement_max=displacement_max
        )

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max, other_peaks = correl.compute_displacements_from_correl(
            c, norm=norm
        )
        displacement_computed = np.array([dx, dy])

        logger.debug(
            k
            + ", displacement = [0, 0]\t error= {}\n".format(
                abs(displacement_computed)
            )
        )

        self.assertTrue(np.allclose([0, 0], displacement_computed, atol=1e-03))

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        logger.debug(
            k
            + ", displacement = {}\t error= {}\n".format(
                self.displacements,
                abs(displacement_computed - self.displacements),
            )
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel.test_correl_square_image_" + k + " = _test")


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

        nb_particles = (max(nx, ny) // 4) ** 2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.
        )


for k, cls in classes.items():

    def _test1(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )
        displacement_computed = np.array([dx, dy])

        logger.debug(
            k
            + ", displacement = [0, 0]\t error= {}\n".format(
                abs(displacement_computed)
            )
        )

        self.assertTrue(np.allclose([0, 0], displacement_computed, atol=1e-03))

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        logger.debug(
            k
            + ", displacement = {}\t error= {}\n".format(
                self.displacements,
                abs(displacement_computed - self.displacements),
            )
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel1.test_correl_rectangular_image_" + k + " = _test1")


class TestCorrel2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx0 = 48
        ny0 = 96
        nx1 = 32
        ny1 = 64

        # Sometimes errors with nx = ny = 16
        # nx = 16
        # ny = 16

        displacement_x = 3.3
        displacement_y = 5.8

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (max(nx0, ny0) // 4) ** 2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements,
            nb_particles,
            shape_im0=(ny0, nx0),
            shape_im1=(ny1, nx1),
            epsilon=0.,
        )

        cls.im1 = cls.im1.astype("float32")


for k, cls in classes2.items():

    def _test2(self, cls=cls, k=k):
        correl = cls(self.im0.shape, self.im1.shape, mode="valid")

        # with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        logger.debug(
            k
            + ", displacement = {}\t error= {}\n".format(
                self.displacements,
                abs(displacement_computed - self.displacements),
            )
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel2.test_correl_images_diff_sizes" + k + " = _test2")

if __name__ == "__main__":
    unittest.main()
