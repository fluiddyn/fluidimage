"""Correlation classes
======================

The correlation classes are able to compute correlations with
different methods.

.. autoclass:: CorrelBase
   :members:
   :private-members:

.. autoclass:: CorrelScipySignal
   :members:
   :private-members:

.. autoclass:: CorrelScipyNdimage
   :members:
   :private-members:

.. autoclass:: CorrelPythran
   :members:
   :private-members:

.. autoclass:: CorrelPyCuda
   :members:
   :private-members:

.. autoclass:: CorrelTheano
   :members:
   :private-members:

.. autoclass:: CorrelFFTNumpy
   :members:
   :private-members:

.. autoclass:: CorrelFFTW
   :members:
   :private-members:

.. autoclass:: CorrelCuFFT
   :members:
   :private-members:

.. autoclass:: CorrelSKCuFFT
   :members:
   :private-members:

"""

from __future__ import division, print_function

import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import correlate
from numpy.fft import fft2, ifft2

from .fft import FFTW2DReal2Complex, CUFFT2DReal2Complex, SKCUFFT2DReal2Complex

from .correl_pythran import correl_pythran

from .correl_pycuda import correl_pycuda

from .errors import PIVError
from .subpix import SubPix

try:
    import theano
except ImportError:
    pass


class CorrelBase(object):
    """This class is meant to be subclassed, not instantiated directly."""
    _tag = 'base'

    def __init__(self, im0_shape, im1_shape, method_subpix='centroid',
                 nsubpix=1):
        self.inds0 = tuple(np.array(im0_shape)//2 - 1)
        self.subpix = SubPix(method=method_subpix, nsubpix=nsubpix)

    def compute_displacement_from_indices(self, indices):
        """Compute the displacement from a couple of indices."""
        return self.inds0[1] - indices[0], self.inds0[0] - indices[1]

    def compute_displacement_from_correl(self, correl, coef_norm=1.,
                                         method_subpix=None, nsubpix=None):

        """Compute the displacement (with subpix) from a correlation."""
        iy, ix = np.unravel_index(correl.argmax(), correl.shape)
        correl_max = correl[iy, ix]/coef_norm
        try:
            indices = self.subpix.compute_subpix(
                correl, ix, iy, method_subpix, nsubpix=nsubpix)
        except PIVError as e:
            indices = ix, iy
            dx, dy = self.compute_displacement_from_indices(indices)
            e.results_compute_displacement_from_correl = (
                dx, dy, correl_max)
            raise e
        dx, dy = self.compute_displacement_from_indices(indices)
        return dx, dy, correl_max


class CorrelPythran(CorrelBase):
    """Correlation using pythran.
       Correlation class by hands with with numpy.
    """
    _tag = 'pythran'

    def __init__(self, im0_shape, im1_shape=None,
                 method_subpix='centroid', nsubpix=1, displacement_max=None,
                 mode=None):

        if displacement_max is None:
            if im0_shape == im1_shape:
                displacement_max = min(im0_shape) // 2 - 1
            else:
                displacement_max = min(im0_shape[0]-im1_shape[0],
                                       im0_shape[1]-im1_shape[1]) // 2 - 1
        if displacement_max <= 0:
            raise ValueError(
                'displacement_max <= 0 : problem with images shapes')
        self.displacement_max = displacement_max
        super(CorrelPythran, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        ind0x = displacement_max
        ind0y = displacement_max

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        return correl_pythran(im0, im1, self.displacement_max)


class CorrelPyCuda(CorrelBase):
    """Correlation using pycuda.
       Correlation class by hands with with cuda.
    """
    _tag = 'pycuda'

    def __init__(self, im0_shape, im1_shape=None,
                 method_subpix='centroid', nsubpix=1, displacement_max=None,
                 mode=None):

        if displacement_max is None:
            if im0_shape == im1_shape:
                displacement_max = max(im0_shape) // 2 #min(max(im0_shape), max(im1_shape)) // 2
            else:
                displacement_max = max(im0_shape[0]-im1_shape[0],
                                       im0_shape[1]-im1_shape[1]) // 2 - 1
        if displacement_max <= 0:
            raise ValueError(
                'displacement_max <= 0 : problem with images shapes')

        self.displacement_max = displacement_max

        super(CorrelPyCuda, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        ind0x = displacement_max
        ind0y = displacement_max

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        return correl_pycuda(im0, im1, self.displacement_max)


class CorrelScipySignal(CorrelBase):
    """Correlations using scipy.signal.correlate2d"""
    _tag = 'scipy.signal'

    def __init__(self, im0_shape, im1_shape=None,
                 method_subpix='centroid', nsubpix=1, mode='same'):

        if im1_shape is None:
            im1_shape = im0_shape

        super(CorrelScipySignal, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        modes = ['valid', 'same']
        if mode not in modes:
            raise ValueError('mode should be in ' + modes)
        self.mode = mode
        if mode == 'same':
            ny, nx = im0_shape
            if nx % 2 == 0:
                ind0x = nx // 2 - 1
            else:
                ind0x = nx // 2
            if ny % 2 == 0:
                ind0y = ny // 2 - 1
            else:
                ind0y = ny // 2

        else:
            ny, nx = np.array(im0_shape) - np.array(im1_shape)
            ind0x = nx // 2
            ind0y = ny // 2

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        if self.mode == 'valid':
            correl = correlate2d(im0, im1, mode='valid')
        elif self.mode == 'same':
            correl = correlate2d(im0, im1, mode='same', fillvalue=im1.min())
        else:
            assert False, 'Bad value for self.mode'

        return correl, norm


class CorrelScipyNdimage(CorrelBase):
    """Correlations using scipy.ndimage.correlate."""
    _tag = 'scipy.ndimage'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid',
                 nsubpix=1):
        super(CorrelScipyNdimage, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)
        self.inds0 = tuple(np.array(im0_shape)//2)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        return correlate(im0, im1, mode='constant', cval=im1.min()), norm


class CorrelTheano(CorrelBase):
    """Correlations using theano.tensor.nnet.conv2d"""
    _tag = 'theano'

    def __init__(self, im0_shape, im1_shape=None,
                 mode='disp', method_subpix='centroid', nsubpix=1,
                 displacement_max=None):
        if im1_shape is None:
            im1_shape = im0_shape

        if displacement_max is None:
            if im0_shape == im1_shape:
                displacement_max = max(im0_shape) // 2 #min(max(im0_shape), max(im1_shape)) // 2
            else:
                displacement_max = max(im0_shape[0]-im1_shape[0],
                                       im0_shape[1]-im1_shape[1]) // 2 - 1
        if displacement_max <= 0:
            raise ValueError(
                'displacement_max <= 0 : problem with images shapes')

        super(CorrelTheano, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        modes = ['valid', 'same', 'disp']
        if mode not in modes:
            raise ValueError('mode should be in ' + modes)
        self.mode = mode
        self.ny0, self.nx0 = im0_shape
        self.ny1, self.nx1 = im1_shape
        self.displacement_max = displacement_max
        if mode == 'same':
            self.ny, self.nx = im0_shape
            if self.nx % 2 == 0:
                ind0x = self.nx // 2 - 1
            else:
                ind0x = self.nx // 2
            if self.ny % 2 == 0:
                ind0y = self.ny // 2 - 1
            else:
                ind0y = self.ny // 2

        elif mode == 'valid':
            self.ny, self.nx = np.array(im0_shape) - np.array(im1_shape) + 1
            ind0x = self.nx // 2
            ind0y = self.ny // 2
        else:
            self.ny = displacement_max*2+1
            self.nx = self.ny
            ind0x = displacement_max
            ind0y = displacement_max

        im00 = theano.tensor.tensor4("im00", dtype='float32')
        im11 = theano.tensor.tensor4("im11", dtype='float32')
        modec = theano.compile.get_default_mode()
        # modec = modec.including('conv_meta')
        if mode == 'same':
            correl_theano = theano.tensor.nnet.conv2d(
                im00, im11,
                image_shape=(1, 1, 2*self.ny0-1, 2*self.nx0-1),
                filter_shape=(1, 1, )+im1_shape,
                border_mode='valid')
        elif mode == 'valid':
            correl_theano = theano.tensor.nnet.conv2d(
                im00, im11,
                input_shape=(1, 1, )+im0_shape,
                filter_shape=(1, 1, )+im1_shape,
                border_mode='valid')
        else:
            if ((self.ny0 <= 2*self.displacement_max + self.ny1) &
                    (self.nx0 <= 2*self.displacement_max + self.nx1)):
                correl_theano = theano.tensor.nnet.conv2d(
                    im00, im11,
                    input_shape=(1, 1, 2*displacement_max+self.ny1,
                                 2*displacement_max+self.nx1),
                    filter_shape=(1, 1, )+im1_shape,
                    border_mode='valid')
            elif ((self.ny0 > 2*self.displacement_max + self.ny1) &
                    (self.nx0 > 2*self.displacement_max + self.nx1)):
                correl_theano = theano.tensor.nnet.conv2d(
                    im00, im11,
                    image_shape=(1, 1, )+im0_shape,
                    filter_shape=(1, 1, )+im1_shape,
                    border_mode='valid')
            else:
                assert False, 'Bad value for self.mode'

        self.correlf = theano.function(inputs=[im00, im11],
                                       outputs=[correl_theano], mode=modec)

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        im1 = np.rot90(im1, 2)
        im1 = im1.reshape(1, 1, self.ny1, self.nx1)
        if self.mode == 'valid':
            im0 = im0.reshape(1, 1, self.ny0, self.nx0)
        elif self.mode == 'same':
            im0b = im1.min() * np.ones((2*self.ny-1, 2*self.nx-1),
                                       dtype=np.float32)
            im0b[self.ny//2-1:self.ny+self.ny//2-1,
                 self.nx//2-1:self.nx+self.nx//2-1] = im0
            # Correlation with periodic condition (==FFT version) :
            # im0 = np.tile(im0, (3, 3))
            # im0 = im0[self.nx//2+1:2*self.nx+self.nx//2,
            #           self.ny//2+1:2*self.ny+self.ny//2]
            im0 = im0b.reshape(1, 1, 2*self.ny-1, 2*self.nx-1)
        elif self.mode == 'disp':
            if ((self.ny0 < 2*self.displacement_max + self.ny1) &
                    (self.nx0 < 2*self.displacement_max + self.nx1)):

                    im0b = np.zeros((2*self.displacement_max + self.ny1,
                                     2*self.displacement_max + self.nx1),
                                    dtype=np.float32)
                    i00 = (2*self.displacement_max + self.nx1 - self.nx0) // 2
                    j00 = (2*self.displacement_max + self.ny1 - self.ny0) // 2
                    im0b[j00:self.ny0+j00, i00:self.nx0+i00] = im0
                    im0 = im0b.reshape(1, 1,
                                       2*self.displacement_max + self.ny1,
                                       2*self.displacement_max + self.nx1)
            elif ((self.ny0 > 2*self.displacement_max + self.ny1) &
                    (self.nx0 > 2*self.displacement_max + self.nx1)):
                    im0 = im0.reshape(1, 1, self.ny0, self.nx0)
        else:
            assert False, 'Bad value for self.mode'

        correl = self.correlf(im0, im1)
        correl = np.asarray(correl)
        if ((self.ny0 > 2*self.displacement_max + self.ny1) &
                (self.nx0 > 2*self.displacement_max + self.nx1) &
                (self.mode == 'disp')):
            i00 = (self.nx0 - self.nx1 + 1) // 2 - self.displacement_max
            j00 = (self.ny0 - self.ny1 + 1) // 2 - self.displacement_max
            correl = correl[0, 0, 0, j00:j00+2*self.displacement_max+1,
                            i00:i00+2*self.displacement_max+1]
        else:
            correl = correl.reshape(self.ny, self.nx)
        return correl, norm


class CorrelFFTNumpy(CorrelBase):
    """Correlations using numpy.fft."""
    _tag = 'np.fft'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid',
                 nsubpix=1):
        super(CorrelFFTNumpy, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('with this correlation method the input images '
                             'have to have the same shape.')

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        corr = ifft2(fft2(im0).conj() * fft2(im1)).real
        return np.fft.fftshift(corr[::-1, ::-1]), norm


class CorrelFFTW(CorrelBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""
    FFTClass = FFTW2DReal2Complex
    _tag = 'fftw'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid',
                 nsubpix=1):
        super(CorrelFFTW, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('with this correlation method the input images '
                             'have to have the same shape.')

        n0, n1 = im1_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1))
        return np.fft.fftshift(corr[::-1, ::-1]), norm


class CorrelCuFFT(CorrelBase):
    _tag = 'cufft'
    """Correlations using fluidimage.fft.CUFFT2DReal2Complex"""
    FFTClass = CUFFT2DReal2Complex

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid',
                 nsubpix=1):
        super(CorrelCuFFT, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('with this correlation method the input images '
                             'have to have the same shape.')

        n0, n1 = im1_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1)).real * im0.size**2
        return np.fft.fftshift(corr[::-1, ::-1]), norm


class CorrelSKCuFFT(CorrelBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""
    FFTClass = SKCUFFT2DReal2Complex
    _tag = 'skcufft'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid',
                 nsubpix=1):
        super(CorrelSKCuFFT, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix, nsubpix=nsubpix)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('with this correlation method the input images '
                             'have to have the same shape.')

        n0, n1 = im1_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1))
        return np.fft.fftshift(corr[::-1, ::-1]), norm


correlation_classes = {
    v._tag: v for k, v in locals().items()
    if k.startswith('Correl') and not k.endswith('Base')}
