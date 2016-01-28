import unittest
import numpy as np

from correl import FFTW2DReal2Complex, CUFFT2DReal2Complex


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

    def compute_and_check2(self, func, op):

        energyX = op.compute_energy_from_spatial(func)
        print energyX
        func_fft = op.fft(func)
        energyK = op.compute_energy_from_Fourier(func_fft)
        back = op.ifft(func_fft)
        energyX = op.compute_energy_from_spatial(back)
        func_fft_2 = op.fft(back)
        energyKback = op.compute_energy_from_Fourier(func_fft_2)
        print energyKback
        rtol = 8e-05
        atol = 1e-04
        self.assertTrue(np.allclose(func_fft, func_fft_2,
                                    rtol=rtol, atol=atol))

        # tmp = np.absolute(func - back) - atol - rtol*np.abs(back)

        # print(tmp.max())

        self.assertTrue(np.allclose(func, back, rtol=rtol, atol=atol))

        self.assertAlmostEqual(energyX/energyK, 1., places=3)
        self.assertAlmostEqual(energyK/energyKback, 1., places=3)

    def test_fft_random(self):
        """random"""
        nx = 128
        ny = 64
        op = FFTW2DReal2Complex(nx, ny)

        func = (np.random.random(op.shapeX))

        func = np.array(func, dtype=op.type_real)
        func_fft = op.fft(func)
        func = op.ifft(func_fft)

        self.compute_and_check2(func, op)

        op = CUFFT2DReal2Complex(nx, ny)

        func_fft = op.fft(func)
        func1 = op.ifft(func_fft)

        self.compute_and_check2(func1, op)

if __name__ == '__main__':
    unittest.main()
