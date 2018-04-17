# TO DO
#####
# ADD epsilon, reynolds stresses etc...


import numpy as np


def compute_grid(xs, ys, deltaxs, deltays):

    x = np.unique(xs)
    y = np.unique(ys)
    X, Y = np.meshgrid(x, y)
    X = X.transpose()
    Y = Y.transpose()
    U = np.reshape(deltaxs, (x.size, y.size))
    V = np.reshape(deltays, (x.size, y.size))

    X = X
    Y = Y
    dx = X[1][0] - X[0][0]
    dy = Y[0][1] - Y[0][0]
    U = U
    V = V
    return X, Y, dx, dy, U, V


def compute_derivatives(dx, dy, U, V, edge_order=2):

    dUdx = np.gradient(U, dx, edge_order=edge_order)[0]
    dUdy = np.gradient(U, dy, edge_order=edge_order)[1]
    dVdx = np.gradient(V, dx, edge_order=edge_order)[0]
    dVdy = np.gradient(V, dy, edge_order=edge_order)[1]

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


def oneD_fourier_transform(x, signal, axis=0, parseval=False):

    """
    Computes the 1D Fourier Transform

    INPUT:
    x: 1D np.array
    signal: np.array to Fourier transform
    axis: direction of the fourier transform
    parseval=True  -> chexk parseval theorem

    OUTPUT:
    ft = fourier transform
    omega = puslation (in rad/s if x in s)
    psd: power spectral density normalized such that
    np.sum(signal**2) * dx / Lx = np.sum(psd) * domega

    """

    n = x.size
    dx = x[1] - x[0]
    Lx = np.max(x) - np.min(x)

    ft = np.fft.fftshift(np.fft.fft(signal, axis=axis), axes=axis)
    omega = np.fft.fftshift(np.fft.fftfreq(n, dx)) * 2 * np.pi

    Lomega = np.max(omega) - np.min(omega)
    domega = omega[1] - omega[0]

    psd = 1.0 / Lomega / n * np.abs(ft) ** 2

    if parseval:
        print("np.sum(signal**2) * dx / Lx =")
        print(np.sum(signal ** 2) * dx / Lx)
        print("np.sum(psd) * domega=")
        print(np.sum(psd) * domega)

    return ft, omega, psd


def twoD_fourier_transform(X, Y, U, axis=(1, 2), parseval=False):
    """
    Computes the 2D Fourier Transform
    INPUT:
    X: 2D np.array
    Y: 2D np.array
    U: np.array to Fourier transform
    axis: directions of the fourier transform
    parseval=True  -> chexk parseval theorem

    OUTPUT:
    ft = fourier transform
    kx: puslation (in rad/m if X in m)
    ky: pulsation (in rad/m if Y in m)
    psd: power spectral density normalized such that
    np.sum(signal**2) * dx / Lx = np.sum(psd) * domega
    """
    nx = X.shape[0]
    ny = X.shape[1]
    dx = X[1][0] - X[0][0]
    dy = Y[0][1] - Y[0][0]
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)

    ft = np.fft.fftshift(np.fft.fft2(U, axes=axis), axes=axis)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, dx)) * (2 * np.pi)  # in rad/m
    ky = np.fft.fftshift(np.fft.fftfreq(ny, dy)) * (2 * np.pi)  # in rad/m

    Lkx = np.max(kx) - np.min(kx)
    dkx = kx[1] - kx[0]
    Lky = np.max(ky) - np.min(ky)
    dky = ky[1] - ky[0]
    # Kx, Ky = np.meshgrid(kx, ky)
    # Kx = Kx.transpose()
    # Ky = Ky.transpose()

    psd = 1.0 / Lkx / nx / Lky / ny * np.abs(ft) ** 2

    if parseval:
        print("np.sum(signal**2)* dx* dy/ Lx/ Ly")
        print(np.sum(np.power(U, 2) * 1.0 * dx * dy / Lx / Ly))
        print("np.sum(psd) *dkx * dky =")
        print(np.sum(psd) * 1.0 * dkx * dky)

    return ft, kx, ky, psd
