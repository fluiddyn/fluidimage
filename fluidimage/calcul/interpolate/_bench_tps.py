
from time import clock

import resource

import numpy as np

import matplotlib.pyplot as plt

from fluidimage.calcul.interpolate.thin_plate_spline import (
    compute_tps_coeff,
    ThinPlateSpline,
)


def get_memory_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.


def bench(n0, n1):
    # set of random x coordinates from 0 to 2pi
    x = 2 * np.pi * np.random.rand(n0)
    y = 2 * np.pi * np.random.rand(n0)

    U = x

    # calculate tps coeff
    centers = np.vstack([x, y])

    t = clock()
    U_smooth, U_tps = compute_tps_coeff(centers, U, 0)
    t_tps_coeff = clock() - t

    # interpolation grid
    xI = yI = np.linspace(0, 2 * np.pi, n1)
    XI, YI = np.meshgrid(xI, yI)
    XI = XI.ravel()
    YI = YI.ravel()

    new_positions = np.vstack([XI, YI])

    # evaluate interpolation on the new grid
    t = clock()
    tps = ThinPlateSpline(new_positions, centers)
    memory = get_memory_usage()
    t_init = clock() - t

    t = clock()
    U_eval = tps.compute_field(U_tps)
    t_eval = clock() - t

    t = clock()
    DUX_eval, DUY_eval = tps.compute_gradient(U_tps)
    t_eval_grad = clock() - t

    return t_tps_coeff, t_init, t_eval, t_eval_grad, memory


ns = np.array([10, 20, 40, 60, 80, 100, 140, 200, 400, 600, 800, 1000])

t_tps_coeff = np.empty(ns.shape)
t_init = np.empty_like(t_tps_coeff)
t_eval = np.empty_like(t_tps_coeff)
t_eval_grad = np.empty_like(t_tps_coeff)
memory = np.empty_like(t_tps_coeff)

n0 = 20

for i, n1 in enumerate(ns):
    a, b, c, d, memory[i] = bench(n0, n1)
    t_tps_coeff[i] = a
    t_init[i] = b
    t_eval[i] = c
    t_eval_grad[i] = d


plt.figure()
ax = plt.gca()

ax.plot(ns, t_tps_coeff / ns, "b")
ax.plot(ns, t_init / ns, "r")
ax.plot(ns, t_eval / ns, "g")
ax.plot(ns, t_eval_grad / ns, "y")


plt.figure()
ax = plt.gca()

ax.plot(ns, memory, "r")

ax.set_ylabel("memory (Mo)")

plt.show()
