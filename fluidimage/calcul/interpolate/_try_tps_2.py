import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from fluidimage.calcul.interpolate.thin_plate_spline2 import ThinPlateSpline


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


# interpolation grid

xI = yI = np.arange(0, 2 * pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

tps = ThinPlateSpline(
    new_positions,
    centers,
    U,
    subdom_size,
    smoothing_coef,
    threshold=1,
    pourc_buffer_area=0.5,
)

U_eval = tps.compute_U_eval()
DMX_eval, DMY_eval = tps.compute_dxy_eval()


myplot(1, XI, YI, U_eval)

myplot(2, XI, YI, DMX_eval)
myplot(3, XI, YI, DMY_eval)


plt.show()
