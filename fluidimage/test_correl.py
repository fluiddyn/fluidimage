import unittest
import numpy as np

from correl import FFTW2DReal2Complex


class TestFFTW2DReal2Complex(unittest.TestCase):

    def test_fft(self):
        """simple"""
        nx = 4
        ny = 2
        op = FFTW2DReal2Complex(nx, ny)

        func_fft = np.zeros(op.shapeK, dtype=op.type_complex)
        func_fft[0, 1] = 1

        self.compute_and_check(func_fft, op)

    def compute_and_check(self, func_fft, op):

        energyK = op.compute_energy_from_Fourier(func_fft)

        func = op.ifft(func_fft)
        energyX = op.compute_energy_from_spatial(func)

        back_fft = op.fft(func)
        energyKback = op.compute_energy_from_Fourier(back_fft)
        back = op.ifft(back_fft)

        rtol = 8e-05
        atol = 1e-04
        self.assertTrue(np.allclose(func_fft, back_fft, rtol=rtol, atol=atol))

        # tmp = np.absolute(func - back) - atol - rtol*np.abs(back)

        # print(tmp.max())

        self.assertTrue(np.allclose(func, back, rtol=rtol, atol=atol))

        self.assertAlmostEqual(energyX/energyK, 1., places=3)
        self.assertAlmostEqual(energyK/energyKback, 1., places=3)

    def test_fft_random(self):
        """random"""
        nx = 64
        ny = 128
        op = FFTW2DReal2Complex(nx, ny)

        func_fft = (np.random.random(op.shapeK) +
                    1.j*np.random.random(op.shapeK))
        func_fft = np.array(func_fft, dtype=op.type_complex)
        func = op.ifft(func_fft)
        func_fft = op.fft(func)

        self.compute_and_check(func_fft, op)


if __name__ == '__main__':
    unittest.main()
