import numpy as np
import pytest
from numpy.random import PCG64, Generator

from fluidimage.calcul.fft import classes


@pytest.mark.parametrize("cls", classes)
def test_fft_random(cls):

    nx, ny = 20, 24
    oper = cls(nx, ny)

    generator = Generator(PCG64())

    arr = generator.random(nx * ny, dtype=oper.type_real).reshape(oper.shapeX)
    arr_fft = oper.fft(arr)

    energyX = oper.compute_energy_from_spatial(arr)
    energyK = oper.compute_energy_from_Fourier(arr_fft)
    back = oper.ifft(arr_fft) / oper.coef_norm
    energyX = oper.compute_energy_from_spatial(back)
    arr_fft_2 = oper.fft(back)
    energyKback = oper.compute_energy_from_Fourier(arr_fft_2)
    rtol = 8e-05
    atol = 1e-04
    assert np.allclose(arr_fft, arr_fft_2, rtol=rtol, atol=atol)
    assert np.allclose(arr, back, rtol=rtol, atol=atol)
    assert energyK == pytest.approx(energyKback)
    assert energyX == pytest.approx(energyK)

    correl = oper.ifft(arr_fft.conj() * arr_fft)
    assert np.allclose(correl.max() / (2 * energyK * oper.coef_norm_correl), 1)


@pytest.mark.parametrize("cls", classes)
def test_fft_simple(cls):
    """simple"""
    nx = 4
    ny = 2
    oper = cls(nx, ny)

    arr_fft = np.zeros(oper.shapeK, dtype=cls.type_complex)
    arr_fft[0, 1] = 1

    energyK = oper.compute_energy_from_Fourier(arr_fft)

    func = oper.ifft(arr_fft)
    energyX = oper.compute_energy_from_spatial(func)

    back_fft = oper.fft(func) / oper.coef_norm
    back = oper.ifft(back_fft)

    rtol = 8e-05
    atol = 1e-04
    assert np.allclose(arr_fft, back_fft, rtol=rtol, atol=atol)
    assert np.allclose(func, back, rtol=rtol, atol=atol)

    assert energyX, pytest.approx(energyK)

    energyKback = oper.compute_energy_from_Fourier(back_fft)

    assert energyK, pytest.approx(energyKback)
