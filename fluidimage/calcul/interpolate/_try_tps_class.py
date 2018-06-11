import numpy as np

import scipy as sc

from matplotlib.mlab import griddata
from scipy import interpolate
from numpy import pi

import matplotlib.pyplot as plt

plt.ion()

from fluidimage.calcul.interpolate.thin_plate_spline import (
    compute_tps_coeff,
    ThinPlateSpline,
)

n0 = 100
n1 = 100

# method = 'griddata_scipy'
method = "tps"


def myplot(i, x, y, U, title=None):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)

    if title is not None:
        ax.set_title(title)


# set of random x coordinates from 0 to 2pi
x = 2 * np.pi * np.random.rand(n0)
y = 2 * np.pi * np.random.rand(n0)

U = np.exp(-((x - pi) ** 2 + (y - pi) ** 2))  # gaussian
# U = x  # linear

myplot(0, x, y, U, "input data")

centers = np.vstack([x, y])
# interpolation grid
xI = yI = np.linspace(0, 2 * pi, n1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

if method == "tps":

    # calculate tps coeff
    U_smooth, U_tps = compute_tps_coeff(centers, U, 0)
    # evaluate interpolation on the new grid
    tps = ThinPlateSpline(new_positions, centers)

    U_eval = tps.compute_field(U_tps)

    # U_eval = griddata(x,y,U,xI,yI,'linear') #griddata method from matplotlib

    DUX_eval, DUY_eval = tps.compute_gradient(U_tps)

elif method == "griddata_mpl":

    U_eval = griddata(x, y, U, xI, yI, "linear")

    DUX_eval, DUY_eval = sc.gradient(U_eval)

elif method == "griddata_scipy":

    grid_x, grid_y = np.meshgrid(xI, yI)
    # nearest, linear, cubic
    U_eval = interpolate.griddata(centers.T, U, (grid_x, grid_y), "cubic", 0)

    DUX_eval, DUY_eval = sc.gradient(U_eval)


myplot(1, XI, YI, U_eval, "U_eval")

myplot(2, XI, YI, DUX_eval, "DUX_eval")
myplot(3, XI, YI, DUY_eval, "DUY_eval")

plt.show()
