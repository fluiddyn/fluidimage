import logging
import unittest

import numpy as np

from fluidimage.calcul.correl import (
    CorrelFFTBase,
    CorrelPyCuda,
    CorrelPythran,
    CorrelScipySignal,
    _like_fftshift,
    correlation_classes,
)
from fluidimage.synthetic import make_synthetic_images

# config_logging('debug')
logger = logging.getLogger("fluidimage")

classes = {k.replace(".", "_"): v for k, v in correlation_classes.items()}
classes_real_space = {
    "signal": CorrelScipySignal,
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
    classes_real_space.pop("pycuda")

try:
    import skcuda
except ImportError:
    classes.pop("skcufft")


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
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.0
        )


for method, cls in classes.items():

    def _test(self, cls=cls, name=method):
        if issubclass(cls, CorrelFFTBase):
            displacement_max = "50%"
        else:
            displacement_max = None

        correl = cls(
            self.im0.shape, self.im1.shape, displacement_max=displacement_max
        )

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        (
            dx,
            dy,
            correl_max,
            other_peaks,
        ) = correl.compute_displacements_from_correl(c, norm=norm)
        displacement_computed = np.array([dx, dy])

        print(f"{name}, displacement = [0, 0]\t{abs(displacement_computed) = }")

        assert np.allclose([0, 0], displacement_computed, atol=1e-03)
        assert np.allclose(correl_max, 1.0), correl_max

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        print(
            f"{name}, displacement = {self.displacements}\t"
            f"error = {abs(displacement_computed - self.displacements)}"
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel.test_correl_square_image_" + method + " = _test")


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
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.0
        )


for method, cls in classes.items():

    def _test1(self, cls=cls, name=method):
        correl = cls(self.im0.shape, self.im1.shape)

        # first, no displacement
        c, norm = correl(self.im0, self.im0)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )
        displacement_computed = np.array([dx, dy])

        logger.debug(
            name
            + ", displacement = [0, 0]\t error= {}\n".format(
                abs(displacement_computed)
            )
        )

        assert np.allclose(correl_max, 1.0), correl_max
        assert np.allclose([0, 0], displacement_computed, atol=1e-03)

        # then, with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        print(
            f"{name}, displacement = {self.displacements}\t"
            f"error = {abs(displacement_computed - self.displacements)}"
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel1.test_correl_rectangular_image_" + method + " = _test1")


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
            epsilon=0.0,
        )

        cls.im1 = cls.im1.astype("float32")


for method, cls in classes_real_space.items():

    def _test2(self, cls=cls, name=method):
        correl = cls(self.im0.shape, self.im1.shape, mode="valid")

        # with the 2 figures with displacements
        c, norm = correl(self.im0, self.im1)
        dx, dy, correl_max, _ = correl.compute_displacements_from_correl(
            c, norm=norm
        )

        displacement_computed = np.array([dx, dy])

        print(
            f"{name}, displacement = {self.displacements}\t"
            f"error = {abs(displacement_computed - self.displacements)}"
        )

        self.assertTrue(
            np.allclose(self.displacements, displacement_computed, atol=0.8)
        )

    exec("TestCorrel2.test_correl_images_diff_sizes_" + method + " = _test2")


def _test_like_fftshift(n0, n1):
    correl = np.reshape(np.arange(n0 * n1, dtype=np.float32), (n0, n1))
    assert np.allclose(
        _like_fftshift(correl),
        np.ascontiguousarray(np.fft.fftshift(correl[::-1, ::-1])),
    )


def test_like_fftshift():
    _test_like_fftshift(24, 32)
    _test_like_fftshift(21, 32)
    _test_like_fftshift(12, 13)
    _test_like_fftshift(7, 9)


if __name__ == "__main__":
    unittest.main()
