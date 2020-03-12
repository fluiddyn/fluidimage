import numpy as np

from .util import (
    compute_1dspectrum,
    compute_2dspectrum,
)


def test_compute_1dspectrum():
    nx = 100
    lx = 10.0
    dx = lx / nx
    x = dx * np.arange(nx)
    signal = np.random.rand(nx)

    energy = np.mean(signal ** 2) / 2

    signal_fft, omega, psd = compute_1dspectrum(x, signal)

    energy_fft = 0.5 * np.sum(np.abs(signal_fft) ** 2)
    assert np.allclose(energy, energy_fft), (energy, energy_fft)
    domega = omega[1] - omega[0]
    assert np.allclose(domega, 2 * np.pi / lx)
    assert np.allclose(energy, np.sum(psd) * domega)


def _test_compute_2dspectrum(nt):

    nx = 10
    ny = 100
    lx = 1000
    ly = 10000
    dx = lx / nx
    dy = ly / ny
    x = dx * np.arange(nx)
    y = dy * np.arange(ny)

    X, Y = np.meshgrid(x, y)

    assert X.shape == (ny, nx)

    nt = None

    if nt:
        axes = (1, 2)
        size = X.size * nt
        shape = [nt, *X.shape]
    else:
        nt = 1
        axes = (0, 1)
        size = X.size
        shape = X.shape

    signal = np.random.rand(size).reshape(shape)

    energy = 0.5 * np.mean(signal ** 2)

    signal_fft, kx, ky, psd = compute_2dspectrum(X, Y, signal, axes=axes)
    energy_fft = 0.5 * np.sum(np.abs(signal_fft) ** 2) / nt
    assert np.allclose(energy, energy_fft), (energy, energy_fft)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]

    energy_psd = dkx * dky * np.sum(psd) / nt
    assert np.allclose(energy, energy_psd)


def test_compute_2dspectrum_input2d():
    _test_compute_2dspectrum(nt=None)


def test_compute_2dspectrum_input3d():
    _test_compute_2dspectrum(nt=4)
