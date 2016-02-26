import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from tps_base_flat import (
    compute_tps_coeff, compute_tps_matrix, compute_tps_matrices_dxy)


x = 2*np.pi*np.random.randn(100)  # set of random x coordinates from 0 to 2pi
y = 4*np.pi*np.random.randn(100)  # set of random y coordinates

centers = np.vstack([x, y])

U = np.exp(-((x-pi)**2 + (y-pi)**2))  # gaussian
U = x  # linear

# calculate tps coeff
U_smooth, U_tps = compute_tps_coeff(centers, U, 0)

# interpolation grid
xI = np.arange(0, 2*pi, 0.1)
yI = np.arange(0, 4*pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
npy, npx = XI.shape
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

# evaluate interpolation on the new grid
EM = compute_tps_matrix(new_positions, centers)

U_eval = np.dot(U_tps, EM)
U_eval = U_eval.reshape([npy, npx])


def myimshow(ax, U):
    ax.imshow(U, interpolation='none', vmin=-1, vmax=1)


plt.figure(1)
ax = plt.gca()
myimshow(ax, U_eval)

DMX, DMY = compute_tps_matrices_dxy(new_positions, centers)
DUX_eval = np.dot(U_tps, DMX)
DUY_eval = np.dot(U_tps, DMY)
DUX_eval = DUX_eval.reshape([npy, npx])
DUY_eval = DUY_eval.reshape([npy, npx])


plt.figure(2)
ax = plt.gca()
myimshow(ax, DUX_eval)

plt.figure(3)
ax = plt.gca()
myimshow(ax, DUY_eval)

plt.show()
