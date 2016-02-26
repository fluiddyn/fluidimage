import numpy as np

from numpy import pi

import matplotlib.pyplot as plt

from tps_base import tps_coeff, tps_eval_dxy, tps_eval, tps_eval_T


x = 2*np.pi*np.random.randn(100)  # set of random x coordinates from 0 to 2pi
y = 2*np.pi*np.random.randn(100)  # set of random y coordinates

# to get things like in Matlab
x = x.reshape([x.size, 1])
y = y.reshape([y.size, 1])
centers = np.hstack([x, y])

U = np.exp(-((x-pi)**2 + (y-pi)**2))  # gaussian
U = x  # linear

# calculate tps coeff
U_smooth, U_tps = tps_coeff(centers, U, 0)

# interpolation grid
xI = yI = np.arange(0, 2*pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
npy, npx = XI.shape
XI = XI.ravel().reshape([XI.size, 1])
YI = YI.ravel().reshape([YI.size, 1])

new_positions = np.hstack([XI, YI])

# evaluate interpolation on the new grid
EM = tps_eval(new_positions, centers)

EM_T = tps_eval_T(new_positions, centers)

assert np.allclose(EM.T, EM_T)

U_eval = np.dot(EM, U_tps)
U_eval = U_eval.reshape([npy, npx])


def myimshow(ax, U):
    ax.imshow(U, interpolation='none', vmin=-1, vmax=1)


plt.figure(1)
ax = plt.gca()
myimshow(ax, U_eval)
# ax.plot(yI, U_eval[:,5])

DMX, DMY = tps_eval_dxy(new_positions, centers)
DUX_eval = DMX.dot(U_tps)
DUY_eval = DMY.dot(U_tps)
DUX_eval = DUX_eval.reshape([npy, npx])
DUY_eval = DUY_eval.reshape([npy, npx])


plt.figure(2)
ax = plt.gca()
myimshow(ax, DUX_eval)
# ax.plot(xI, DU_eval[:,5])

plt.figure(3)
ax = plt.gca()
myimshow(ax, DUY_eval)

# plt.figure(4)
# ax = plt.gca()
# ax.plot(x, U_eval[50,:], x, DUX_eval[50,:], x, DUY_eval[50,:])
# ax.plot(xI, U_eval[50, :], xI, DUX_eval[50, :], xI, DUY_eval[50, :])


plt.show()
