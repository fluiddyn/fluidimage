# TO DO
#####
# ADD epsilon, reynolds stresses etc...


import numpy as np


def get_grid_from_ivecs_final(ixvecs_final, iyvecs_final):
    x = np.unique(ixvecs_final)
    y = np.unique(iyvecs_final)
    X, Y = np.meshgrid(x, y)
    return X, Y


def reshape_on_grid_final(ixvecs_final, iyvecs_final, deltaxs, deltays):
    X, Y = get_grid_from_ivecs_final(ixvecs_final, iyvecs_final)
    shape = X.shape
    if ixvecs_final[1] == ixvecs_final[0]:
        second_index_corresponds_to_x = True
        shape = shape[::-1]
    else:
        second_index_corresponds_to_x = False

    U = np.reshape(deltaxs, shape)
    V = np.reshape(deltays, shape)

    if second_index_corresponds_to_x:
        U = U.T
        V = V.T

    return X, Y, U, V


def compute_grid(xs, ys, deltaxs, deltays):
    x = np.unique(xs)
    y = np.unique(ys)
    X, Y = np.meshgrid(x, y)
    U = np.reshape(deltaxs, X.shape)
    V = np.reshape(deltays, X.shape)
    return X, Y, U, V


def compute_derivatives(dx, dy, U, V, edge_order=2):
    dUdx, dUdy = np.gradient(U, dx, dy, edge_order=edge_order)
    dVdx, dVdy = np.gradient(V, dx, dy, edge_order=edge_order)
    return dUdx, dUdy, dVdx, dVdy


def compute_rot(dUdy, dVdx):
    rot = dVdx - dUdy
    return rot


def compute_div(dUdx, dVdy):
    div = dUdx + dVdy
    return div


def compute_ken(U, V):
    ken = (U ** 2 + V ** 2) / 2
    return ken


def compute_norm(U, V):
    norm = np.sqrt(U ** 2 + V ** 2)
    return norm


def compute_1dspectrum(x, signal, axis=0):

    """
    Computes the 1D Fourier Transform

    INPUT:
    x: 1D np.array
    signal: np.array to Fourier transform
    axis: direction of the Fourier transform

    OUTPUT:
    signal_fft = fourier transform
    omega = puslation (in rad/s if x in s)
    psd: power spectral density normalized such that
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
