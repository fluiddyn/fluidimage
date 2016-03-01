import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from fluidimage.calcul.interpolate.thin_plate_spline import (
    compute_tps_coeff, ThinPlateSpline)

n0 = 100
n1 = 100


def myplot(i, x, y, U):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)


# set of random x coordinates from 0 to 2pi
x = 2*np.pi*np.random.rand(n0)
y = 2*np.pi*np.random.rand(n0)

U = np.exp(-((x-pi)**2 + (y-pi)**2))  # gaussian
# U = x  # linear

myplot(0, x, y, U)


# calculate tps coeff
centers = np.vstack([x, y])
U_smooth, U_tps = compute_tps_coeff(centers, U, 0)

# interpolation grid
xI = yI = np.linspace(0, 2*pi, n1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

# evaluate interpolation on the new grid
tps = ThinPlateSpline(new_positions, centers)

U_eval = tps.compute_field(U_tps)

myplot(1, XI, YI, U_eval)

DUX_eval, DUY_eval = tps.compute_gradient(U_tps)

myplot(2, XI, YI, DUX_eval)
myplot(3, XI, YI, DUY_eval)

plt.show()
