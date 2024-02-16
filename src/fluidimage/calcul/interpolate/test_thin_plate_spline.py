import numpy as np

from .thin_plate_spline import (
    ThinPlateSpline,
    ThinPlateSplineNumpy,
    compute_tps_coeff,
)

pi = np.pi


def test_tps():
    n0 = 10
    n1 = 10

    # set of random x coordinates from 0 to 2pi
    x = 2 * pi * np.random.rand(n0)
    y = 2 * pi * np.random.rand(n0)

    U = np.exp(-((x - pi) ** 2 + (y - pi) ** 2))  # gaussian

    centers = np.vstack([x, y])
    # interpolation grid
    xI = yI = np.linspace(0, 2 * pi, n1)
    XI, YI = np.meshgrid(xI, yI)
    XI = XI.ravel()
    YI = YI.ravel()

    new_positions = np.vstack([XI, YI])

    # calculate tps coeff
    U_smooth, U_tps = compute_tps_coeff(centers, U, 0)
    # evaluate interpolation on the new grid
    tps = ThinPlateSpline(new_positions, centers)
    tps.compute_field(U_tps)
    tps.compute_gradient(U_tps)

    ThinPlateSplineNumpy(new_positions, centers)
