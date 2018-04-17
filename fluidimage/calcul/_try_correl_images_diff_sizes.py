
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

plt.ion()

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import CorrelScipySignal


nx = 64
ny = 64

nx1 = 32
ny1 = 32

displacement_x = 2.
displacement_y = 2.

displacements = np.array([displacement_y, displacement_x])

nb_particles = (nx // 3) ** 2


print("nx: {} ; ny: {} ; nx1: {} ; ny1: {}".format(nx, ny, nx1, ny1))

im0, im1 = make_synthetic_images(
    displacements,
    nb_particles,
    shape_im0=(ny, nx),
    shape_im1=(ny1, nx1),
    epsilon=0.,
)

# plt.figure()

# ax0 = plt.subplot(121)
# ax1 = plt.subplot(122)

# axi0 = ax0.imshow(im0, interpolation='nearest')
# axi1 = ax1.imshow(im1, interpolation='nearest')


classes = {"sig": CorrelScipySignal}


cs = {}
funcs = {}
for k, cls in classes.items():
    calcul_corr = cls(im0.shape, im1.shape, mode="valid")
    funcs[k] = calcul_corr
    cs[k], norm = calcul_corr(im0, im1)


for k, c in cs.items():
    func = funcs[k]
    inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
    if not np.allclose(
        displacements.astype("int"),
        func.compute_displacement_from_indices(inds_max),
    ):
        print(
            "do not understand " + k,
            displacements.astype("int"),
            func.compute_displacement_from_indices(inds_max),
        )

c = cs["sig"]
# plt.figure()

# ax = plt.gca()
# ax.imshow(c, interpolation='nearest')

# plt.show()
