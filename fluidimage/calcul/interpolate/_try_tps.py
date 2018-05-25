import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from fluidimage.calcul.interpolate.thin_plate_spline import (
    compute_tps_coeff_iter,
    compute_tps_matrix,
    compute_tps_matrices_dxy,
)


def myplot(i, x, y, U):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)


# set of random x coordinates from 0 to 2pi
x = 2 * np.pi * np.random.rand(100)
y = 2 * np.pi * np.random.rand(100)

U = np.exp(-((x - pi) ** 2 + (y - pi) ** 2))  # gaussian
# U = x  # linear

myplot(0, x, y, U)


# calculate tps coeff
centers = np.vstack([x, y])
smoothing_coef = 0
subdom_size = 20
U_smooth, U_tps = compute_tps_coeff_iter(centers, U, smoothing_coef, threshold=1)

# interpolation grid
xI = yI = np.arange(0, 2 * pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

# evaluate interpolation on the new grid
EM = compute_tps_matrix(new_positions, centers)

U_eval = np.dot(U_tps, EM)

myplot(1, XI, YI, U_eval)

DMX, DMY = compute_tps_matrices_dxy(new_positions, centers)
DUX_eval = np.dot(U_tps, DMX)
DUY_eval = np.dot(U_tps, DMY)

myplot(2, XI, YI, DUX_eval)
myplot(3, XI, YI, DUY_eval)

plt.show()
