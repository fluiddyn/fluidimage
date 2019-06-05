
from time import clock

import numpy as np

from fluidimage.calcul.correl import (
    CorrelPyCuda,
    CorrelPythran,
    CorrelScipySignal,
    CorrelTheano,
)
from fluidimage.synthetic import make_synthetic_images

nx = 64
ny = 64

nx1 = 32
ny1 = 32

displacement_x = 2.
displacement_y = 2.

displacements = np.array([displacement_y, displacement_x])

nb_particles = (nx // 3)**2


print(f'nx: {nx} ; ny: {ny} ; nx1: {nx1} ; ny1: {ny1}')

im0, im1 = make_synthetic_images(
    displacements, nb_particles, shape_im0=(ny, nx), shape_im1=(ny1, nx1),
    epsilon=0.)

im1 = im1.astype('float32')

classes = {'sig': CorrelScipySignal,
           'theano': CorrelTheano,
           'pycuda': CorrelPyCuda,
           'pythran': CorrelPythran}


cs = {}
funcs = {}
for k, cls in classes.items():
    calcul_corr = cls(im0.shape, im1.shape, mode='valid')
    funcs[k] = calcul_corr
    t = clock()
    cs[k] = calcul_corr(im0, im1)
    print('calcul correl with {} : {} s'.format(k, clock() - t))
