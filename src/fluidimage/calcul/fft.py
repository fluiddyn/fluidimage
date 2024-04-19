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

from abc import ABC, abstractmethod

import numpy as np
import numpy.fft as np_fft
from transonic import Array, Type, boost

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


A2d_complex = Array[Type(np.complex64, np.complex128), "2d"]


@boost
def _compute_energy_from_fourier(field_fft: A2d_complex, coef_norm: int):
    """Simple Pythran implementation of

    (
        0.5 / coef_norm
        * (
            np.sum(abs(field_fft[:, 0]) ** 2 + abs(field_fft[:, -1]) ** 2)
            + 2 * np.sum(abs(field_fft[:, 1:-1]) ** 2)
        )
    )
    """
    n0, n1 = field_fft.shape
    result = 0.0
    for i0 in range(n0):
        result += abs(field_fft[i0, 0]) ** 2 + abs(field_fft[i0, n1 - 1]) ** 2
        for i1 in range(1, n1 - 1):
            result += 2 * abs(field_fft[i0, i1]) ** 2
    return 0.5 / coef_norm * result


class OperatorFFTBase(ABC):
    """Abstract class for FFT operators"""

    coef_norm: int
    type_real: str = "float32"
    type_complex: str = "complex64"

    def __init__(self, nx, ny):
        shapeX = [ny, nx]
        shapeK = [ny, nx // 2 + 1]

        self.shapeX = shapeX
        self.shapeK = shapeK

        self.coef_norm = nx * ny
        self.coef_norm_correl = self.coef_norm**2
        self.coef_norm_energy = self.coef_norm**2

    @abstractmethod
    def fft(self, field):
        """Forwards Fast Fourier Transform"""

    @abstractmethod
    def ifft(self, field_fft):
        """Inverse Fast Fourier Transform"""

    def compute_energy_from_fourier(self, field_fft):
        """Compute the energy from a field in Fourier space"""
        return _compute_energy_from_fourier(field_fft, self.coef_norm_energy)

    def compute_energy_from_spatial(self, field):
        """Compute energy from a field in real space"""
        return np.mean(abs(field) ** 2) / 2

    def project_fft_on_real(self, field_fft):
        """Project a field in Fourier space onto the real manifold"""
        return self.fft(self.ifft(field_fft))


class CUFFT2DReal2Complex(OperatorFFTBase):
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

    def fft(self, field):
        arr_dev = self.thr.to_device(field.astype(self.type_complex))
        self.fftplan(arr_dev, arr_dev, 1.0 / self.coef_norm)
        return arr_dev.get()

    def ifft(self, field_fft):
        arr_dev = self.thr.to_device(field_fft)
        self.fftplan(arr_dev, arr_dev, self.coef_norm, inverse=True)
        return arr_dev.get()

    def compute_energy_from_fourier(self, field_fft):
        return np.sum(abs(field_fft) ** 2) / 2 / self.coef_norm_energy


class CUFFT2DReal2ComplexFloat64(CUFFT2DReal2Complex):
    """A class to use cufft with float64."""

    type_real = "float64"
    type_complex = "complex128"


class SKCUFFT2DReal2Complex(OperatorFFTBase):
    """A class to use skcuda-cufft with float32."""

    type_real = "float32"
    type_complex = "complex64"

    def __init__(self, nx, ny):
        super().__init__(nx, ny)
        self.fftplan = skfft.Plan(self.shapeX, np.float32, np.complex64)
        self.ifftplan = skfft.Plan(self.shapeX, np.complex64, np.float32)

    def fft(self, field):
        x_gpu = gpuarray.to_gpu(field)
        xf_gpu = gpuarray.empty(self.shapeK, np.complex64)
        skfft.fft(x_gpu, xf_gpu, self.fftplan, False)
        return xf_gpu.get()

    def ifft(self, field_fft):
        xf_gpu = gpuarray.to_gpu(field_fft)
        x_gpu = gpuarray.empty(self.shapeX, np.float32)
        skfft.ifft(xf_gpu, x_gpu, self.ifftplan, False)
        return x_gpu.get()


class FFTW2DReal2Complex(OperatorFFTBase):
    """A class to use fftw with float32.

    These ffts are NOT normalized (faster)!

    """

    type_real = "float32"
    type_complex = "complex64"

    def __init__(self, nx, ny):
        super().__init__(nx, ny)

        self.arrayX = pyfftw.empty_aligned(self.shapeX, self.type_real)
        self.arrayK = pyfftw.empty_aligned(self.shapeK, self.type_complex)

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

    def fft(self, field):
        self.arrayX[:] = field
        self.fftplan()
        return self.arrayK.copy()

    def ifft(self, field_fft):
        self.arrayK[:] = field_fft
        self.ifftplan(normalise_idft=False)
        return self.arrayX.copy()


class FFTW2DReal2ComplexFloat64(FFTW2DReal2Complex):
    """A class to use fftw with float64."""

    type_real = "float64"
    type_complex = "complex128"


class NumpyFFT2DReal2Complex(OperatorFFTBase):
    """FFT operator using numpy.fft"""

    def __init__(self, nx, ny):
        super().__init__(nx, ny)
        self.coef_norm_correl = self.coef_norm
        self.coef_norm = 1

    def fft(self, field):
        return np_fft.fft2(field)

    def ifft(self, field_fft):
        return np_fft.ifft2(field_fft)

    def compute_energy_from_fourier(self, field_fft):
        return np.sum(abs(field_fft) ** 2) / 2 / self.coef_norm_energy


class NumpyFFT2DReal2ComplexFloat64(NumpyFFT2DReal2Complex):
    """FFT operator using numpy.fft"""

    type_real = "float64"
    type_complex = "complex128"


classes = [
    FFTW2DReal2Complex,
    FFTW2DReal2ComplexFloat64,
    NumpyFFT2DReal2Complex,
    NumpyFFT2DReal2ComplexFloat64,
]
