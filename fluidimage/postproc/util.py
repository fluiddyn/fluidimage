"""Utilities for post-processing (:mod:`fluidimage.postproc.util`)
==================================================================

.. autofunction:: get_grid_from_ivecs_final

.. autofunction:: reshape_on_grid_final

.. autofunction:: compute_rot

.. autofunction:: compute_div

.. autofunction:: compute_1dspectrum

.. autofunction:: compute_2dspectrum
"""

import numpy as np


def get_grid_from_ivecs_final(x_flat, y_flat):
    """Get a 2d grid from flat arrays"""

    x = np.unique(x_flat)
    y = np.unique(y_flat)

    if x_flat[1] == x_flat[0]:
        assert y_flat[1] != y_flat[0]
        # second_index_corresponds_to_x = True
        dx = x_flat[len(y)] - x_flat[0]
        dy = y_flat[1] - y_flat[0]
    else:
        assert y_flat[1] == y_flat[0]
        # second_index_corresponds_to_x = False
        dx = x_flat[1] - x_flat[0]
        dy = y_flat[len(x)] - y_flat[0]

    if dx < 0:
        x = x[::-1]

    if dy < 0:
        y = y[::-1]

    X, Y = np.meshgrid(x, y)
    return X, Y


def reshape_on_grid_final(x_flat, y_flat, deltaxs, deltays):
    """Reshape flat arrays on a 2d grid"""

    X, Y = get_grid_from_ivecs_final(x_flat, y_flat)
    shape = X.shape
    if x_flat[1] == x_flat[0]:
        assert y_flat[1] != y_flat[0]
        second_index_corresponds_to_x = True
        shape = shape[::-1]
    else:
        assert y_flat[1] == y_flat[0]
        second_index_corresponds_to_x = False

    U = np.reshape(deltaxs, shape)
    V = np.reshape(deltays, shape)

    if second_index_corresponds_to_x:
        U = U.T
        V = V.T

    return X, Y, U, V


def compute_rot(dUdy, dVdx):
    """Compute the rotational"""
    return dVdx - dUdy


def compute_div(dUdx, dVdy):
    """Compute the divergence"""
    return dUdx + dVdy


def compute_1dspectrum(x, signal, axis=0):
    """
    Computes the 1D Fourier Transform

    Parameters
    ----------

    x: 1D np.ndarray

    signal: np.ndarray

    axis: int

      Direction of the Fourier transform

    Returns
    -------

    signal_fft: np.ndarray

      fourier transform

    omega: np.ndarray

      puslation (in rad/s if x in s)

    psd: np.ndarray

      power spectral density normalized such that
      np.sum(signal**2) * dx / Lx = np.sum(psd) * domega

    """
    n = x.size
    dx = x[1] - x[0]
    Lx = n * dx
    dk = 2 * np.pi / Lx

    signal_fft = (
        dx / Lx * np.fft.fftshift(np.fft.fft(signal, axis=axis), axes=axis)
    )
    omega = np.fft.fftshift(np.fft.fftfreq(n, dx)) * 2 * np.pi

    psd = 0.5 / dk * np.abs(signal_fft) ** 2

    return signal_fft, omega, psd


def compute_2dspectrum(X, Y, signal, axes=(1, 2)):
    """
    Computes the 2D Fourier Transform

    INPUT:
    X: 2D np.array
    Y: 2D np.array
    signal: np.array to Fourier transform
    axis: directions of the fourier transform

    OUTPUT:
    signal_fft = fourier transform
    kx: puslation (in rad/m if X in m)
    ky: pulsation (in rad/m if Y in m)
    psd: power spectral density normalized such that
    np.sum(signal**2) * dx / Lx = np.sum(psd) * domega
    """
    if X.ndim == 2:
        nx = X.shape[1]
        ny = X.shape[0]
        x = X[0, :2]
        y = Y[:2, 0]
    else:
        nx = X.size
        x = X
        ny = Y.size
        y = Y

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    lx = nx * dx
    ly = ny * dy
    dkx = 2 * np.pi / lx
    dky = 2 * np.pi / ly

    # energy = 0.5 * np.mean(signal**2)

    signal_fft = (
        1 / (nx * ny) * np.fft.fftshift(np.fft.fft2(signal, axes=axes), axes=axes)
    )

    # if axes == (1, 2):
    #     nt = signal.shape[0]
    # else:
    #     nt = 1

    # energy_fft = 0.5 / nt * np.sum(np.abs(signal_fft)**2)
    # assert np.allclose(energy, energy_fft), (energy, energy_fft)

    # in rad/m
    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * (2 * np.pi)
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * (2 * np.pi)

    psd = 0.5 / (dkx * dky) * np.abs(signal_fft) ** 2

    # energy_psd = dkx * dky * np.sum(psd) / nt
    # assert np.allclose(energy, energy_psd)

    return signal_fft, kx, ky, psd
