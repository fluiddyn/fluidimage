"""

correlation computed with fft are much faster !

"""

from __future__ import division, print_function

import numpy as np
from scipy.signal import convolve2d

from fft import FFTW2DReal2Complex, CUFFT2DReal2Complex

# if 'OMP_NUM_THREADS' in os.environ:
#     nthreads = int(os.environ['OMP_NUM_THREADS'])
# else:
#     pass

# It seems that it is better to used nthreads = 1 for the fft with very small
# dimension used for PIV
nthreads = 1


def calcul_correl_norm_scipy(im0, im1):
    correl = convolve2d(im1, np.rot90(im0, 2), boundary='wrap', mode='same')
    correl_min = correl.min()
    return (correl - correl_min) / (correl.max() - correl_min)


class CorrelWithFFT(object):
    def __init__(self, nx, ny):
        self.op = FFTW2DReal2Complex(nx, ny)
#       self.op = CUFFT2DReal2Complex(nx, ny)

    def calcul_correl_norm(self, im0, im1):
        fft = self.op.fft
        ifft = self.op.ifft
        correl = np.real(np.fft.fftshift(ifft(fft(im0).conj() * fft(im1))))
        correl_min = correl.min()
        return (correl - correl_min) / (correl.max() - correl_min)
