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

.. autoclass:: SubPix
   :members:
   :private-members:

"""

from __future__ import division, print_function

import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import correlate
from numpy.fft import fft2, ifft2

from .fft import FFTW2DReal2Complex, CUFFT2DReal2Complex

try:
    import theano
    from theano.sandbox.cuda import cuda_available, GpuOp
    from theano.sandbox.cuda.fftconv import conv2d_fft
    import scikits.cuda
except ImportError:
    pass


class NoPeakError(Exception):
    """No peak"""


class CorrelBase(object):
    """This class is meant to be subclassed, not instantiated directly."""
    _tag = 'base'

    def __init__(self, im0_shape, im1_shape, method_subpix='centroid'):
        self.inds0 = tuple(np.array(im0_shape)//2 - 1)

        self.subpix = SubPix(method=method_subpix)

    def compute_displacement_from_indices(self, indices):
        """Compute the displacement from a couple of indices."""
        return self.inds0[0] - indices[0], self.inds0[1] - indices[1]

    def compute_displacement_from_correl(
            self, correl, method_subpix=None):
        """Compute the displacement (with subpix) from a correlation."""
        iy, ix = np.unravel_index(correl.argmax(), correl.shape)
        indices = self.subpix.compute_subpix(
            correl, ix, iy, method_subpix)
        return self.compute_displacement_from_indices(indices)


class CorrelScipySignal(CorrelBase):
    """Correlations using scipy.signal.correlate2d"""
    _tag = 'scipy.signal'

    def __init__(self, im0_shape, im1_shape=None,
                 method_subpix='centroid', mode='same'):

        if im1_shape is None:
            im1_shape = im0_shape

        super(CorrelScipySignal, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix)

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

        return correl/norm


class CorrelScipyNdimage(CorrelBase):
    """Correlations using scipy.ndimage.correlate."""
    _tag = 'scipy.ndimage'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid'):
        super(CorrelScipyNdimage, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix)
        self.inds0 = tuple(np.array(im0_shape)//2)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        return correlate(im0, im1, mode='constant', cval=im1.min())/norm


class CorrelTheano(CorrelBase):
    """Correlations using theano.tensor.nnet.conv2d"""
    _tag = 'theano'

    def __init__(self, im0_shape, im1_shape=None,
                 mode='same', method_subpix='centroid'):

        if im1_shape is None:
            im1_shape = im0_shape

        super(CorrelTheano, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix)

        modes = ['valid', 'same']
        if mode not in modes:
            raise ValueError('mode should be in ' + modes)
        self.mode = mode
        self.ny0, self.nx0 = im0_shape
        self.ny1, self.nx1 = im1_shape
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

        else:
            self.ny, self.nx = np.array(im0_shape) - np.array(im1_shape) + 1
            ind0x = self.nx // 2
            ind0y = self.ny // 2

        im00 = theano.tensor.tensor4("im00", dtype='float32')
        im11 = theano.tensor.tensor4("im11", dtype='float32')
        modec = theano.compile.get_default_mode()
        # modec = modec.including('conv_meta')
        if mode == 'same':
            correl_theano = theano.tensor.nnet.conv2d(
                im00, im11,
                image_shape=(1, 1, 2*self.nx0-1, 2*self.ny0-1),
                filter_shape=(1, 1, )+im1_shape,
                border_mode='valid')
        else:
            correl_theano = theano.tensor.nnet.conv2d(
                im00, im11,
                image_shape=(1, 1, )+im0_shape,
                filter_shape=(1, 1, )+im1_shape,
                border_mode='valid')

        self.correlf = theano.function(inputs=[im00, im11],
                                       outputs=[correl_theano], mode=modec)

        self.inds0 = tuple([ind0y, ind0x])

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        im1 = np.rot90(im1, 2)
        im1 = im1.reshape(1, 1, self.nx1, self.ny1)
        if self.mode == 'valid':
            im0 = im0.reshape(1, 1, self.nx0, self.ny0)
        elif self.mode == 'same':
            im0b = im1.min() * np.ones((2*self.nx-1, 2*self.ny-1),
                                       dtype=np.float32)
            im0b[self.nx//2-1:self.nx+self.nx//2-1,
                 self.ny//2-1:self.ny+self.ny//2-1] = im0
            # Correlation with periodic condition (==FFT version) :
            # im0 = np.tile(im0, (3, 3))
            # im0 = im0[self.nx//2+1:2*self.nx+self.nx//2,
            #           self.ny//2+1:2*self.ny+self.ny//2]
            im0 = im0b.reshape(1, 1, 2*self.nx-1, 2*self.ny-1)
        else:
            assert False, 'Bad value for self.mode'

        correl = self.correlf(im0, im1)
        correl = np.asarray(correl)
        correl = correl.reshape(self.nx, self.ny)

        return correl/norm


class CorrelFFTNumpy(CorrelBase):
    """Correlations using numpy.fft."""
    _tag = 'np.fft'

    def __init__(self, im0_shape, im1_shape, method_subpix='centroid'):
        super(CorrelFFTNumpy, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix)
        if im0_shape != im1_shape:
            raise ValueError('The input images have to have the same shape.')

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        corr = ifft2(fft2(im0).conj() * fft2(im1)).real / norm
        return np.fft.fftshift(corr[::-1, ::-1])


class CorrelFFTW(CorrelBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""
    FFTClass = FFTW2DReal2Complex
    _tag = 'fftw'

    def __init__(self, im0_shape, im1_shape=None, method_subpix='centroid'):
        super(CorrelFFTW, self).__init__(
            im0_shape, im1_shape, method_subpix=method_subpix)

        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError('The input images have to have the same shape.')

        n0, n1 = im1_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1)) / norm
        return np.fft.fftshift(corr[::-1, ::-1])


class CorrelCuFFT(CorrelBase):
    _tag = 'cufft'
    """Correlations using fluidimage.fft.CUFFT2DReal2Complex"""
    FFTClass = CUFFT2DReal2Complex


class SubPix(object):
    """Subpixel finder"""
    methods = ['2d_gaussian', 'centroid']

    def __init__(self, method='centroid'):
        self.method = method
        # init for 2d_gaussian method
        self.n_subpix_zoom = 2
        xs = np.arange(2*self.n_subpix_zoom+1, dtype=float)
        ys = np.arange(2*self.n_subpix_zoom+1, dtype=float)
        X, Y = np.meshgrid(xs, ys)
        nx, ny = X.shape
        X = X.ravel()
        Y = Y.ravel()
        M = np.reshape(np.concatenate(
            (X**2, Y**2, X, Y, np.ones(nx*ny))), (5, nx*ny)).T
        self.Minv_subpix = np.linalg.pinv(M)

        # init for centroid method
        self.X_centroid, self.Y_centroid = np.meshgrid(range(3), range(3))

    def compute_subpix(self, correl, ix, iy, method=None):
        """Find peak

        Parameters
        ----------

        correl: numpy.ndarray

          Normalized correlation

        ix: integer

        iy: integer

        method: str {'centroid', '2d_gaussian'}

        Notes
        -----

        The two methods...

        using linalg.solve (buggy?)

        """
        if method is None:
            method = self.method

        if method not in self.methods:
            raise ValueError('method has to be in {}'.format(self.methods))

        n = self.n_subpix_zoom

        ny, nx = correl.shape

        if iy-n < 0 or iy+n+1 > ny or \
           ix-n < 0 or ix+n+1 > nx:
            raise NoPeakError

        if method == '2d_gaussian':

            # crop: possibly buggy!
            correl = correl[iy-n:iy+n+1,
                            ix-n:ix+n+1]

            ny, nx = correl.shape

            assert nx == ny == 2*n + 1

            correl_map = correl.ravel()
            correl_map[correl_map == 0.] = 1e-6

            coef = np.dot(self.Minv_subpix, np.log(correl_map))

            sigmax = 1/np.sqrt(-2*coef[0])
            sigmay = 1/np.sqrt(-2*coef[1])
            X0 = coef[2]*sigmax**2
            Y0 = coef[3]*sigmay**2

            tmp = 2*n + 1
            if X0 > tmp or Y0 > tmp:
                raise NoPeakError

        elif method == 'centroid':

            correl = correl[iy-1:iy+2, ix-1:ix+2]
            ny, nx = correl.shape

            sum_correl = np.sum(correl)

            X0 = np.sum(self.X_centroid * correl) / sum_correl
            Y0 = np.sum(self.Y_centroid * correl) / sum_correl

            if X0 > 2 or Y0 > 2:
                raise NoPeakError

        deplx = X0 - nx/2  # displacement x
        deply = Y0 - ny/2  # displacement y

        return deply + iy + 0.5, deplx + ix + 0.5


correlation_classes = {
    v._tag: v for k, v in locals().items()
    if k.startswith('Correl') and not k.endswith('Base')}
