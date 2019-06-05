# -*- coding: utf-8 -*-

from time import clock

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import correlate
from scipy.signal import correlate2d

from fluidimage.calcul.fft import CUFFT2DReal2Complex, FFTW2DReal2Complex


def corr_full(in0, in1):
    norm = np.sum(in1**2)
    return correlate2d(in0, in1, mode='full')/norm


def corr_full_ndimage(in0, in1):
    norm = np.sum(in1**2)
    return correlate(in0, in1)/norm


def corr_fft_numpy(in0, in1):
    norm = np.sum(in1**2)
    return ((ifft2(fft2(in0).conj() * fft2(in1))).real)[::-1, ::-1]/norm


n0 = 64
n1 = 32
print(f'n0: {n0} ; n1: {n1}')

in0 = np.random.randn(n0, n0).astype('float32')
in1 = np.random.randn(n1, n1).astype('float32')


op = FFTW2DReal2Complex(n0, n0)


def corr_fft(in0, in1):
    norm = np.sum(in1**2)
    return ((op.ifft(op.fft(in0).conj() * op.fft(in1))).real)[::-1, ::-1]/norm


op1 = CUFFT2DReal2Complex(n0, n0)


def corr_cufft(in0, in1):
    norm = np.sum(in1**2)
    return ((op1.ifft(op1.fft(in0).conj() * op1.fft(in1))).real)[::-1, ::-1]/norm


t = clock()
corr_full(in0, in1)
print('corr_full(in0, in1) : \t\t{} s'.format(clock() - t))

t = clock()
corr_full_ndimage(in0, in1)
print('corr_full_ndimage(in0, in1) : \t{} s'.format(clock() - t))

t = clock()
corr_fft(in0, in0)
print('corr_fft(in0, in0) : \t\t{} s'.format(clock() - t))

t = clock()
corr_cufft(in0, in0)
print('corr_cufft(in0, in0) : \t\t{} s'.format(clock() - t))

"""Result bench

n0: 32 ; n1: 16
corr_full(in0, in1) : 		0.002623 s
corr_full_ndimage(in0, in1) : 	0.002647 s
corr_fft(in0, in0) : 		0.000211999999999 s
corr_cufft(in0, in0) : 		0.0035 s


n0: 64 ; n1: 32
corr_full(in0, in1) : 		0.035295 s
corr_full_ndimage(in0, in1) : 	0.03343 s
corr_fft(in0, in0) : 		0.000281 s
corr_cufft(in0, in0) : 		0.003 s

The scipy correlation functions are much slower than the fft method (by a
factor ) and than matlab (by a factor ~ 5).

"""
