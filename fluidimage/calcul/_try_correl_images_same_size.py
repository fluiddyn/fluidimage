
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

plt.ion()

from fluidimage.calcul.correl import (
    CorrelScipySignal,
    CorrelScipyNdimage,
    CorrelFFTNumpy,
    CorrelFFTW,
)

from fluidimage.synthetic import make_synthetic_images

nx = 32
ny = 32
displacement_x = 2.
displacement_y = 2.

displacements = np.array([displacement_x, displacement_y])

nb_particles = (nx // 3) ** 2


print("nx: {} ; ny: {}".format(nx, ny))

im0, im1 = make_synthetic_images(
    displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.
)

# plt.figure()

# ax0 = plt.subplot(121)
# ax1 = plt.subplot(122)

# axi0 = ax0.imshow(im0, interpolation='nearest')
# axi1 = ax1.imshow(im1, interpolation='nearest')


classes = {
    "sig": CorrelScipySignal,
    "ndimage": CorrelScipyNdimage,
    "np.fft": CorrelFFTNumpy,
    "fftw": CorrelFFTW,
}


cs = {}
correls = {}
for k, cls in classes.items():
    print(k, cls)
    correl = cls(im0.shape, im1.shape)
    correls[k] = correl
    c, norm = correl(im0, im1)
    cs[k] = c

    inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
    print(inds_max)

    displacement_computed = correl.compute_displacement_from_indices(inds_max)
    print(displacements.astype("int"), displacement_computed)

    if not np.allclose(displacements.astype("int"), displacement_computed):
        print(
            "do not understand " + k,
            displacements.astype("int"),
            displacement_computed,
        )


# plt.figure()

# i = 1
# for k, c in cs.items():
#     ax1 = plt.subplot(2, 2, i)
#     ax1.imshow(c, interpolation='nearest')
#     i += 1

# plt.show()
