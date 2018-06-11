"""FFT classes
==============

Warning: it is more efficient to use not normalized FFT, so we'll do
that.

.. autoclass:: CUFFT2DReal2Complex
   :members:
   :private-members:

.. autoclass:: CUFFT2DReal2ComplexFloat64
   :members:
   :private-members:

.. autoclass:: SKCUFFT2DReal2Complex
   :members:
   :private-members:

.. autoclass:: FFTW2DReal2Complex
   :members:
   :private-members:

.. autoclass:: FFTW2DReal2ComplexFloat64
   :members:
   :private-members:

"""

from __future__ import division, print_function

import numpy as np

try:
    import pyfftw
except ImportError:
    pass

try:
    from reikna.cluda import any_api, cuda_api, ocl_api
    from reikna.fft import FFT
    from reikna.transformations import mul_param
except ImportError:
    pass

try:
    import pycuda._driver
except ImportError:
    pass
else:
    try:
        import pycuda.autoinit
        import pycuda.gpuarray as gpuarray
        import skcuda.fft as skfft
    except (ImportError, pycuda._driver.RuntimeError):
        pass

# if 'OMP_NUM_THREADS' in os.environ:
#     nthreads = int(os.environ['OMP_NUM_THREADS'])
# else:
#     pass

# It seems that it is better to used nthreads = 1 for the fft with
# small size used for PIV
nthreads = 1


class CUFFT2DReal2Complex(object):
    """A class to use cufft with float32."""

    type_real = "float32"
    type_complex = "complex64"

    def __init__(self, nx, ny):

        shapeX = [ny, nx]
        shapeK = [ny, nx]

        self.shapeX = shapeX
        self.arrayK = np.empty(shapeK, dtype=self.type_complex)

        # Pick the first available GPGPU API and make a Thread on it.
        api = any_api()
        # api = cuda_api()
        # api = ocl_api()
        dev = api.get_platforms()[0].get_devices()
        self.thr = api.Thread.create(dev)
        fft = FFT(self.arrayK, axes=(0, 1))
        scale = mul_param(self.arrayK, np.float)
        fft.parameter.input.connect(
            scale, scale.output, input_prime=scale.input, param=scale.param
        )
        self.fftplan = fft.compile(self.thr, fast_math=True)

        self.coef_norm = nx * ny

    def fft(self, ff):
        arr_dev = self.thr.to_device(ff.astype(self.type_complex))
        self.fftplan(arr_dev, arr_dev, 1. / self.coef_norm)
        return arr_dev.get()

    def ifft(self, ff_fft):
        arr_dev = self.thr.to_device(ff_fft)
        self.fftplan(arr_dev, arr_dev, self.coef_norm, inverse=True)
        return arr_dev.get()

    def compute_energy_from_Fourier(self, ff_fft):
        return np.sum(abs(ff_fft) ** 2) / 2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff) ** 2) / 2

    def project_fft_on_realX(self, ff_fft):
        return self.fft(self.ifft(ff_fft))


class CUFFT2DReal2ComplexFloat64(CUFFT2DReal2Complex):
    """A class to use cufft with float64."""

    type_real = "float64"
    type_complex = "complex128"


class SKCUFFT2DReal2Complex(object):
    """A class to use skcuda-cufft with float32."""

    type_real = "float32"
    type_complex = "complex64"

    def __init__(self, nx, ny):

        shapeX = [ny, nx]
        shapeK = [ny, nx // 2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.fftplan = skfft.Plan(self.shapeX, np.float32, np.complex64)
        self.ifftplan = skfft.Plan(self.shapeX, np.complex64, np.float32)

        self.coef_norm = nx * ny

    def fft(self, ff):
        x_gpu = gpuarray.to_gpu(ff)
        xf_gpu = gpuarray.empty(self.shapeK, np.complex64)
        skfft.fft(x_gpu, xf_gpu, self.fftplan, False)
        return xf_gpu.get()

    def ifft(self, ff_fft):
        xf_gpu = gpuarray.to_gpu(ff_fft)
        x_gpu = gpuarray.empty(self.shapeX, np.float32)
        skfft.ifft(xf_gpu, x_gpu, self.ifftplan, False)
        return x_gpu.get()

    def compute_energy_from_Fourier(self, ff_fft):
        return (
            np.sum(abs(ff_fft[:, 0]) ** 2 + abs(ff_fft[:, -1]) ** 2)
            + 2 * np.sum(abs(ff_fft[:, 1:-1]) ** 2)
        ) / 2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff) ** 2) / 2

    def project_fft_on_realX(self, ff_fft):
        return self.fft(self.ifft(ff_fft))


class FFTW2DReal2Complex(object):
    """A class to use fftw with float32.

    These ffts are NOT normalized (faster)!

    """

    type_real = "float32"
    type_complex = "complex64"

    def __init__(self, nx, ny):

        shapeX = [ny, nx]
        shapeK = [ny, nx // 2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.arrayX = pyfftw.empty_aligned(shapeX, self.type_real)
        self.arrayK = pyfftw.empty_aligned(shapeK, self.type_complex)

        self.fftplan = pyfftw.FFTW(
            input_array=self.arrayX,
            output_array=self.arrayK,
            axes=(0, 1),
            direction="FFTW_FORWARD",
            threads=nthreads,
        )
        self.ifftplan = pyfftw.FFTW(
            input_array=self.arrayK,
            output_array=self.arrayX,
            axes=(0, 1),
            direction="FFTW_BACKWARD",
            threads=nthreads,
        )

        self.coef_norm = nx * ny

    def fft(self, ff):
        self.arrayX[:] = ff
        self.fftplan(normalise_idft=False)
        return self.arrayK.copy()

    def ifft(self, ff_fft):
        self.arrayK[:] = ff_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()

    def compute_energy_from_Fourier(self, ff_fft):
        return (
            np.sum(abs(ff_fft[:, 0]) ** 2 + abs(ff_fft[:, -1]) ** 2)
            + 2 * np.sum(abs(ff_fft[:, 1:-1]) ** 2)
        ) / 2

    def compute_energy_from_spatial(self, ff):
        return np.mean(abs(ff) ** 2) / 2

    def project_fft_on_realX(self, ff_fft):
        return self.fft(self.ifft(ff_fft))


class FFTW2DReal2ComplexFloat64(FFTW2DReal2Complex):
    """A class to use fftw with float64."""

    type_real = "float64"
    type_complex = "complex128"
