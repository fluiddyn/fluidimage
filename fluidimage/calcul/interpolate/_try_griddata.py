from time import clock

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from matplotlib import mlab
from scipy import interpolate

from fluidimage.calcul.interpolate import thin_plate_spline_subdom
from fluidimage.calcul.interpolate.griddata import griddata
from fluidimage.calcul.interpolate.thin_plate_spline import (
    ThinPlateSpline,
    compute_tps_coeff,
)

plt.ion()


n0 = 40
n1 = 100

method = "griddata_scipy"
method = "griddata_mpl"
method = "griddata_fluidimage"
# method = 'tps'
# method = 'tps_subdom'

with_gradient = False


def myplot(i, x, y, U, title=None):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)

    if title is not None:
        ax.set_title(title)


L = 2 * np.pi

x = y = np.linspace(0, L, n0)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()

U = np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2))  # gaussian
# U = x  # linear

myplot(0, x, y, U, "input data")

centers = np.vstack([x, y])
# interpolation grid
xI = yI = np.linspace(0, L, n1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

dx = L / n1

new_positions = np.vstack([XI, YI])

t0 = clock()

if method == "tps":

    # calculate tps coeff
    U_smooth, U_tps = compute_tps_coeff(centers, U, 0)
    # evaluate interpolation on the new grid
    tps = ThinPlateSpline(new_positions, centers)

    U_eval = tps.compute_field(U_tps)

    if with_gradient:
        DUX_eval, DUY_eval = tps.compute_gradient(U_tps)

elif method == "tps_subdom":

    smoothing_coef = 0.5
    subdom_size = 50

    tps = thin_plate_spline_subdom.ThinPlateSplineSubdom(
        centers, subdom_size, smoothing_coef, threshold=1, pourc_buffer_area=0.5
    )

    U_smooth, U_tps = tps.compute_tps_coeff_subdom(U)

    tps.init_with_new_positions(new_positions)
    U_eval = tps.compute_eval(U_tps)

elif method == "griddata_mpl":

    U_eval = mlab.griddata(x, y, U, xI, yI, "linear")

elif method == "griddata_scipy":

    grid_x, grid_y = np.meshgrid(xI, yI)
    # nearest, linear, cubic
    U_eval = interpolate.griddata(centers.T, U, (grid_x, grid_y), "cubic", 0)

elif method == "griddata_fluidimage":
    # grid_x, grid_y = np.meshgrid(xI, yI)
    U_eval = griddata(centers, U, (xI, yI))
else:
    raise ValueError


if method.startswith("griddata") and with_gradient:
    DUY_eval, DUX_eval = sc.gradient(U_eval, dx, dx)

print("done in {} s ({})".format(clock() - t0, method))

myplot(1, XI, YI, U_eval, "U_eval")

if with_gradient:
    myplot(2, XI, YI, DUX_eval, "DUX_eval")
    myplot(3, XI, YI, DUY_eval, "DUY_eval")

plt.show()
