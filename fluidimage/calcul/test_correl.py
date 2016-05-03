
from __future__ import print_function

import unittest

import numpy as np

from fluidimage.synthetic import make_synthetic_images
from fluidimage.calcul.correl import correlation_classes
import pylab

classes = {k.replace('.', '_'): v for k, v in correlation_classes.items()}

try:
    from reikna.cluda import any_api
    api = any_api()
except Exception:
    classes.pop('cufft')

try:
    import pycuda
except ImportError:
    classes.pop('pycuda')

try:
    import skcuda
except ImportError:
    classes.pop('skcufft')

try:
    import theano
except ImportError:
    classes.pop('theano')


class TestCorrel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nx = 64
        ny = 64
        displacement_x = 3.5
        displacement_y = 5.3

        cls.displacements = np.array([displacement_x, displacement_y])

        nb_particles = (nx // 4)**2

        cls.im0, cls.im1 = make_synthetic_images(
            cls.displacements, nb_particles, shape_im0=(ny, nx), epsilon=0.)
        #pylab.imshow(cls.im0)
        #pylab.show()

#method_subpix = '2d_gaussian'
method_subpix = 'centroid'
n_subpix = np.arange(1,10)

err = np.zeros([np.shape(classes.items())[0],n_subpix.size])
temp = np.reshape(classes.items(),[6,2])
leg = temp.T[0]

indn = 0
for nsubpix in n_subpix:
    indk = 0
    for k, cls in classes.items():
        def test(self, cls=cls, k=k, nsubpix=nsubpix, indk=indk, indn=indn):
            global err
            correl = cls(self.im0.shape, self.im1.shape, method_subpix=method_subpix, nsubpix=nsubpix)

            # first, no displacement
            c, norm = correl(self.im0, self.im0)
            dx, dy, correl_max = correl.compute_displacement_from_correl(
                c, coef_norm=norm,
            )
            displacement_computed = np.array([dx, dy])
            #        inds_max = np.array(np.unravel_index(c.argmax(), c.shape))
            #        displacement_computed = correl.compute_displacement_from_indices(
            #            inds_max)

            #self.assertTrue(np.allclose(
            #    [0, 0],
            #    displacement_computed, atol=1e-5))
            #print('\n', k, "[0, 0]", displacement_computed)

            # then, with the 2 figures with displacements
            c, norm = correl(self.im0, self.im1)
            dx, dy, correl_max = correl.compute_displacement_from_correl(
                c, coef_norm=norm,
            )

            displacement_computed = np.array([dx, dy])

            #print('\n', k, self.displacements, np.abs(displacement_computed-self.displacements))
            #self.assertTrue(np.allclose(
            #    self.displacements,
            #    displacement_computed,
            #    atol=0.5))
            #return np.abs(displacement_computed-self.displacements)[0]
            err[indk][indn] =  np.abs(displacement_computed-self.displacements)[0]
        exec('TestCorrel.test_correl_' + k + '_' + str(nsubpix) + ' = test')
        indk += 1
    indn += 1

def plot(err, n_subpix, leg, method_subpix):
    pylab.ion()

    for i, legi in enumerate(leg):
        pylab.plot(n_subpix,err[i],'o')
    pylab.legend(leg)
    pylab.xlabel('nsubpix')
    pylab.ylabel('error in pix')
    pylab.title(method_subpix)
    pylab.xlim([0, np.max(n_subpix)+1])



if __name__ == '__main__':
    unittest.main()
    plot(err, n_subpix, leg, method_subpix)
    
