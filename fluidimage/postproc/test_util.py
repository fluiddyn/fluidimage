import numpy as np

from .util import compute_1dspectrum, compute_2dspectrum, reshape_on_grid_final


def test_compute_1dspectrum():
    nx = 100
    lx = 10.0
    dx = lx / nx
    x = dx * np.arange(nx)
    signal = np.random.rand(nx)

    energy = np.mean(signal**2) / 2

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

    energy = 0.5 * np.mean(signal**2)

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


def test_reshape_on_grid_final():
    x_increasing = np.linspace(-1, 1, 2)
    x_decreasing = np.linspace(2, 1, 2)
    y_increasing = np.linspace(0, 2, 10)
    y_decreasing = np.linspace(2, 0, 10)

    x_grid_ii, y_grid_ii = np.meshgrid(x_increasing, y_increasing)
    x_grid_id, y_grid_id = np.meshgrid(x_increasing, y_decreasing)
    x_grid_dd, y_grid_dd = np.meshgrid(x_decreasing, y_decreasing)

    x_flat_ii, y_flat_ii = x_grid_ii.flatten(), y_grid_ii.flatten()
    x_flat_id, y_flat_id = x_grid_id.flatten(), y_grid_id.flatten()
    x_flat_dd, y_flat_dd = x_grid_dd.flatten(), y_grid_dd.flatten()

    out_x_grid_ii, out_y_grid_ii, plus_ii, minus_ii = reshape_on_grid_final(
        x_flat_ii, y_flat_ii, x_flat_ii + y_flat_ii, x_flat_ii - y_flat_ii
    )

    out_x_grid_id, out_y_grid_id, plus_id, minus_id = reshape_on_grid_final(
        x_flat_id, y_flat_id, x_flat_id + y_flat_id, x_flat_id - y_flat_id
    )

    out_x_grid_dd, out_y_grid_dd, plus_dd, minus_dd = reshape_on_grid_final(
        x_flat_dd, y_flat_dd, x_flat_dd + y_flat_dd, x_flat_dd - y_flat_dd
    )

    assert np.allclose(out_x_grid_ii, x_grid_ii)
    assert np.allclose(out_y_grid_ii, y_grid_ii)
    assert np.allclose(out_x_grid_id, x_grid_id)
    assert np.allclose(out_y_grid_id, y_grid_id)
    assert np.allclose(out_x_grid_dd, x_grid_dd)
    assert np.allclose(out_y_grid_dd, y_grid_dd)

    for iy in range(y_increasing.size):
        for ix in range(x_increasing.size):
            assert (
                out_x_grid_ii[iy, ix] + out_y_grid_ii[iy, ix] == plus_ii[iy, ix]
            )
            assert (
                out_x_grid_ii[iy, ix] - out_y_grid_ii[iy, ix] == minus_ii[iy, ix]
            )
            assert (
                out_x_grid_id[iy, ix] + out_y_grid_id[iy, ix] == plus_id[iy, ix]
            )
            assert (
                out_x_grid_id[iy, ix] - out_y_grid_id[iy, ix] == minus_id[iy, ix]
            )
            assert (
                out_x_grid_dd[iy, ix] + out_y_grid_dd[iy, ix] == plus_dd[iy, ix]
            )
            assert (
                out_x_grid_dd[iy, ix] - out_y_grid_dd[iy, ix] == minus_dd[iy, ix]
            )
