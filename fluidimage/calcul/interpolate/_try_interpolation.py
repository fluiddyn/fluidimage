
import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from scipy.interpolate import Rbf


def myplot(i, x, y, U):
    plt.figure(i)
    ax = plt.gca()

    ax.scatter(x, y, c=U, vmin=-1, vmax=1)


def my_multiplot(nf, x, y, U, n, func):
    plt.figure(nf)
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        ax.scatter(x, y, c=U[:, i], vmin=-1, vmax=1)
        ax.set_title(func[i])


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

U_vrai = np.exp(-((XI - pi) ** 2 + (YI - pi) ** 2))

func = (
    "linear",
    "inverse",
    "gaussian",
    "multiquadric",
    "cubic",
    "quintic",
    "thin-plate",
)
n = len(func)
ERR_eval = np.zeros((np.size(XI), n))
for i in range(n):
    print(i)
    rbfi = Rbf(x, y, U, function=func[i])
    ERR_eval[:, i] = (rbfi(XI, YI) - U_vrai) / U_vrai

my_multiplot(1, XI, YI, ERR_eval, n, func)

plt.show()
