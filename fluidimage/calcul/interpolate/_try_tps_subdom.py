
import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from fluidimage.calcul.interpolate.thin_plate_spline_subdom import (
    ThinPlateSplineSubdom
)


def myplot(i, x, y, U, title=None):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)

    if title is not None:
        ax.set_title(title)


# set of random x coordinates from 0 to 2pi
x = 2 * np.pi * np.random.rand(100)
y = 2 * np.pi * np.random.rand(100)

U = np.exp(-((x - pi) ** 2 + (y - pi) ** 2))
V = np.exp(-((x - pi / 2) ** 2 + (y - pi / 2) ** 2))

myplot(0, x, y, U, "input U")
myplot(1, x, y, V, "input V")

# calculate tps coeff
centers = np.vstack([x, y])
smoothing_coef = 0
subdom_size = 20

tps = ThinPlateSplineSubdom(
    centers, subdom_size, smoothing_coef, threshold=1, pourc_buffer_area=0.5
)

U_smooth, U_tps = tps.compute_tps_coeff_subdom(U)
V_smooth, V_tps = tps.compute_tps_coeff_subdom(V)

# interpolation grid
xI = yI = np.arange(0, 2 * pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

tps.init_with_new_positions(new_positions)

U_eval = tps.compute_eval(U_tps)
V_eval = tps.compute_eval(V_tps)

myplot(2, XI, YI, U_eval, "U_eval")
myplot(3, XI, YI, V_eval, "V_eval")

plt.show()
