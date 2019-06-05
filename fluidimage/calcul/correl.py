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

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import correlate
from scipy.signal import correlate2d

from transonic import boost

from .correl_pycuda import correl_pycuda
from .errors import PIVError
from .fft import CUFFT2DReal2Complex, FFTW2DReal2Complex, SKCUFFT2DReal2Complex
from .subpix import SubPix

try:
    import theano
except ImportError:
    pass


def compute_indices_from_displacement(dx, dy, indices_no_displ):
    return indices_no_displ[1] - dx, indices_no_displ[0] - dy


def parse_displacement_max(displ_max, im0_shape):
    if isinstance(displ_max, str) and displ_max.endswith("%"):
        return float(displ_max[:-1]) / 100 * max(im0_shape)

    else:
        return displ_max


def _compute_indices_max(correl, norm):

    iy, ix = np.unravel_index(np.nanargmax(correl), correl.shape)

    if norm == 0:
        # I hope it is ok (Pierre)
        correl_max = 0.0
    else:
        correl_max = correl[iy, ix] / norm

    if (
        iy == 0
        or iy == correl.shape[0] - 1
        or ix == 0
        or ix == correl.shape[1] - 1
    ):
        error = PIVError(explanation="Correlation peak touching boundary.")
        error.results = (ix, iy, correl_max)
        raise error

    if np.isnan(np.sum(correl[iy - 1 : iy + 2 : 2, ix - 1 : ix + 2 : 2])):
        error = PIVError(explanation="Correlation peak touching nan.")
        error.results = (ix, iy, correl_max)
        raise error

    return ix, iy, correl_max


class CorrelBase:
    """This class is meant to be subclassed, not instantiated directly."""

    _tag = "base"

    def __init__(
        self,
        im0_shape,
        im1_shape,
        method_subpix="centroid",
        nsubpix=1,
        displacement_max=None,
        particle_radius=3,
        nb_peaks_to_search=1,
        mode=None,
    ):

        self.mode = mode

        self.subpix = SubPix(method=method_subpix, nsubpix=nsubpix)

        self.im0_shape = im0_shape
        self.im1_shape = im1_shape
        self.iy0, self.ix0 = (i // 2 - 1 for i in im0_shape)

        self.displacement_max = parse_displacement_max(
            displacement_max, im0_shape
        )

        self.particle_radius = particle_radius
        self.nb_peaks_to_search = nb_peaks_to_search

        self._init2()

    def _init2(self):
        pass

    def compute_displacement_from_indices(self, ix, iy):
        """Compute the displacement from a couple of indices."""
        return self.ix0 - ix, self.iy0 - iy

    def compute_indices_from_displacement(self, dx, dy):
        return self.ix0 - dx, self.iy0 - dy

    def get_indices_no_displacement(self):
        return self.iy0, self.ix0

    def compute_displacements_from_correl(self, correl, norm=1.0):
        """Compute the displacement from a correlation."""

        try:
            ix, iy, correl_max = _compute_indices_max(correl, norm)
        except PIVError as e:
            ix, iy, correl_max = e.results
            # second chance to find a better peak...
            correl[
                iy - self.particle_radius : iy + self.particle_radius + 1,
                ix - self.particle_radius : ix + self.particle_radius + 1,
            ] = np.nan
            try:
                ix2, iy2, correl_max2 = _compute_indices_max(correl, norm)
            except PIVError as e2:
                dx, dy = self.compute_displacement_from_indices(ix, iy)
                e.results = (dx, dy, correl_max)
                raise e

            else:
                ix, iy, correl_max = ix2, iy2, correl_max2

        dx, dy = self.compute_displacement_from_indices(ix, iy)

        if self.nb_peaks_to_search == 1:
            other_peaks = None
        elif self.nb_peaks_to_search >= 1:
            other_peaks = []
            for ip in range(0, self.nb_peaks_to_search - 1):
                correl[
                    iy - self.particle_radius : iy + self.particle_radius + 1,
                    ix - self.particle_radius : ix + self.particle_radius + 1,
                ] = np.nan
                try:
                    ix, iy, correl_max_other = _compute_indices_max(correl, norm)
                except PIVError:
                    break

                dx_other, dy_other = self.compute_displacement_from_indices(
                    ix, iy
                )
                other_peaks.append((dx_other, dy_other, correl_max_other))
        # print('found {} peaks'.format(len(other_peaks) + 1))
        else:
            raise ValueError

        return dx, dy, correl_max, other_peaks

    def apply_subpix(self, dx, dy, correl):
        """Compute the displacement with the subpix method."""
        ix, iy = self.compute_indices_from_displacement(dx, dy)
        ix, iy = self.subpix.compute_subpix(correl, ix, iy)
        return self.compute_displacement_from_indices(ix, iy)


A = "float32[][]"


@boost
def correl_numpy(im0: A, im1: A, disp_max: int):
    """Correlations by hand using only numpy.

    Parameters
    ----------

    im0, im1 : images
      input images : 2D matrix

    disp_max : int
      displacement max.

    Notes
    -------

    im1_shape inf to im0_shape

    Returns
    -------

    the computing correlation (size of computed correlation = disp_max*2 + 1)

    """
    norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2))

    ny = nx = int(disp_max) * 2 + 1
    ny0, nx0 = im0.shape
    ny1, nx1 = im1.shape

    zero = np.float32(0.0)
    correl = np.empty((ny, nx), dtype=np.float32)

    for xiy in range(disp_max + 1):
        dispy = -disp_max + xiy
        nymax = ny1 + min(ny0 // 2 - ny1 // 2 + dispy, 0)
        ny1dep = -min(ny0 // 2 - ny1 // 2 + dispy, 0)
        ny0dep = max(0, ny0 // 2 - ny1 // 2 + dispy)
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = nx1 + min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx1dep = -min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx0dep = max(0, nx0 // 2 - nx1 // 2 + dispx)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy, xix] = tmp / (nxmax * nymax)
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy, xix + disp_max + 1] = tmp / (nxmax * nymax)
    for xiy in range(disp_max):
        dispy = xiy + 1
        nymax = ny1 - max(ny0 // 2 + ny1 // 2 + dispy - ny0, 0)
        ny1dep = 0
        ny0dep = ny0 // 2 - ny1 // 2 + dispy
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = nx1 + min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx1dep = -min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx0dep = max(0, nx0 // 2 - nx1 // 2 + dispx)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy + disp_max + 1, xix] = tmp / (nxmax * nymax)
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy + disp_max + 1, xix + disp_max + 1] = tmp / (nxmax * nymax)
    correl = correl * im1.size
    return correl, norm


class CorrelPythran(CorrelBase):
    """Correlation using pythran.
       Correlation class by hands with with numpy.
    """

    _tag = "pythran"

    def _init2(self):

        if self.displacement_max is None:
            if self.im0_shape == self.im1_shape:
                displacement_max = min(self.im0_shape) // 2 - 1
            else:
                displacement_max = (
                    min(
                        self.im0_shape[0] - self.im1_shape[0],
                        self.im0_shape[1] - self.im1_shape[1],
                    )
                    // 2
                    - 1
                )
        if displacement_max <= 0:
            raise ValueError(
                "displacement_max <= 0 : problem with images shapes?"
            )

        self.displacement_max = displacement_max

        self.ix0 = displacement_max
        self.iy0 = displacement_max

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        correl, norm = correl_numpy(im0, im1, self.displacement_max)
        return correl, norm


class CorrelPyCuda(CorrelBase):
    """Correlation using pycuda.
       Correlation class by hands with with cuda.
    """

    _tag = "pycuda"

    def _init2(self):

        if self.displacement_max is None:
            if self.im0_shape == self.im1_shape:
                displacement_max = min(self.im0_shape) // 2
            else:
                displacement_max = (
                    min(
                        self.im0_shape[0] - self.im1_shape[0],
                        self.im0_shape[1] - self.im1_shape[1],
                    )
                    // 2
                    - 1
                )
        if displacement_max <= 0:
            raise ValueError(
                "displacement_max <= 0 : problem with images shapes?"
            )

        self.displacement_max = displacement_max

        self.ix0 = displacement_max
        self.iy0 = displacement_max

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        correl, norm = correl_pycuda(im0, im1, self.displacement_max)
        self._add_info_to_correl(correl)
        return correl, norm


class CorrelScipySignal(CorrelBase):
    """Correlations using scipy.signal.correlate2d"""

    _tag = "scipy.signal"

    def _init2(self):
        if self.mode is None:
            self.mode = "same"

        modes = ["valid", "same"]
        if self.mode not in modes:
            raise ValueError("mode should be in " + modes)

        if self.mode == "same":
            ny, nx = self.im0_shape
            if nx % 2 == 0:
                ind0x = nx // 2 - 1
            else:
                ind0x = nx // 2
            if ny % 2 == 0:
                ind0y = ny // 2 - 1
            else:
                ind0y = ny // 2

        else:
            ny, nx = np.array(self.im0_shape) - np.array(self.im1_shape)
            ind0x = nx // 2
            ind0y = ny // 2

        self.iy0, self.ix0 = (ind0y, ind0x)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2))
        if self.mode == "valid":
            correl = correlate2d(im0, im1, mode="valid")
        elif self.mode == "same":
            correl = correlate2d(im0, im1, mode="same", fillvalue=im1.min())
        else:
            assert False, "Bad value for self.mode"

        return correl, norm


class CorrelScipyNdimage(CorrelBase):
    """Correlations using scipy.ndimage.correlate."""

    _tag = "scipy.ndimage"

    def _init2(self):
        self.iy0, self.ix0 = (i // 2 for i in self.im0_shape)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1 ** 2)
        correl = correlate(im0, im1, mode="constant", cval=im1.min())

        return correl, norm


class CorrelTheano(CorrelBase):
    """Correlations using theano.tensor.nnet.conv2d"""

    _tag = "theano"

    def _init2(self):
        if self.mode is None:
            mode = self.mode = "disp"

        im0_shape = self.im0_shape
        im1_shape = self.im1_shape

        if im1_shape is None:
            im1_shape = im0_shape

        if self.displacement_max is None:
            if im0_shape == im1_shape:
                displacement_max = max(im0_shape) // 2
            else:
                displacement_max = (
                    max(im0_shape[0] - im1_shape[0], im0_shape[1] - im1_shape[1])
                    // 2
                    - 1
                )
        if displacement_max <= 0:
            raise ValueError("displacement_max <= 0 : problem with images shapes")

        modes = ["valid", "same", "disp"]
        if mode not in modes:
            raise ValueError("mode should be in " + modes)

        self.mode = mode
        self.ny0, self.nx0 = im0_shape
        self.ny1, self.nx1 = im1_shape
        self.displacement_max = displacement_max
        if mode == "same":
            self.ny, self.nx = im0_shape
            if self.nx % 2 == 0:
                ind0x = self.nx // 2 - 1
            else:
                ind0x = self.nx // 2
            if self.ny % 2 == 0:
                ind0y = self.ny // 2 - 1
            else:
                ind0y = self.ny // 2

        elif mode == "valid":
            self.ny, self.nx = np.array(im0_shape) - np.array(im1_shape) + 1
            ind0x = self.nx // 2
            ind0y = self.ny // 2
        else:
            self.ny = displacement_max * 2 + 1
            self.nx = self.ny
            ind0x = displacement_max
            ind0y = displacement_max

        im00 = theano.tensor.tensor4("im00", dtype="float32")
        im11 = theano.tensor.tensor4("im11", dtype="float32")
        modec = theano.compile.get_default_mode()
        # modec = modec.including('conv_meta')
        if mode == "same":
            correl_theano = theano.tensor.nnet.conv2d(
                im00,
                im11,
                image_shape=(1, 1, 2 * self.ny0 - 1, 2 * self.nx0 - 1),
                filter_shape=(1, 1) + im1_shape,
                border_mode="valid",
            )
        elif mode == "valid":
            correl_theano = theano.tensor.nnet.conv2d(
                im00,
                im11,
                input_shape=(1, 1) + im0_shape,
                filter_shape=(1, 1) + im1_shape,
                border_mode="valid",
            )
        else:
            if (self.ny0 <= 2 * self.displacement_max + self.ny1) & (
                self.nx0 <= 2 * self.displacement_max + self.nx1
            ):
                correl_theano = theano.tensor.nnet.conv2d(
                    im00,
                    im11,
                    input_shape=(
                        1,
                        1,
                        2 * displacement_max + self.ny1,
                        2 * displacement_max + self.nx1,
                    ),
                    filter_shape=(1, 1) + im1_shape,
                    border_mode="valid",
                )
            elif (self.ny0 > 2 * self.displacement_max + self.ny1) & (
                self.nx0 > 2 * self.displacement_max + self.nx1
            ):
                correl_theano = theano.tensor.nnet.conv2d(
                    im00,
                    im11,
                    image_shape=(1, 1) + im0_shape,
                    filter_shape=(1, 1) + im1_shape,
                    border_mode="valid",
                )
            else:
                assert False, "Bad value for self.mode"

        self.correlf = theano.function(
            inputs=[im00, im11], outputs=[correl_theano], mode=modec
        )

        self.iy0, self.ix0 = (ind0y, ind0x)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2))
        im1 = np.rot90(im1, 2)
        im1 = im1.reshape(1, 1, self.ny1, self.nx1)
        if self.mode == "valid":
            im0 = im0.reshape(1, 1, self.ny0, self.nx0)
        elif self.mode == "same":
            im0b = im1.min() * np.ones(
                (2 * self.ny - 1, 2 * self.nx - 1), dtype=np.float32
            )
            im0b[
                self.ny // 2 - 1 : self.ny + self.ny // 2 - 1,
                self.nx // 2 - 1 : self.nx + self.nx // 2 - 1,
            ] = im0
            # Correlation with periodic condition (==FFT version) :
            # im0 = np.tile(im0, (3, 3))
            # im0 = im0[self.nx//2+1:2*self.nx+self.nx//2,
            #           self.ny//2+1:2*self.ny+self.ny//2]
            im0 = im0b.reshape(1, 1, 2 * self.ny - 1, 2 * self.nx - 1)
        elif self.mode == "disp":
            if (self.ny0 < 2 * self.displacement_max + self.ny1) & (
                self.nx0 < 2 * self.displacement_max + self.nx1
            ):

                im0b = np.zeros(
                    (
                        2 * self.displacement_max + self.ny1,
                        2 * self.displacement_max + self.nx1,
                    ),
                    dtype=np.float32,
                )
                i00 = (2 * self.displacement_max + self.nx1 - self.nx0) // 2
                j00 = (2 * self.displacement_max + self.ny1 - self.ny0) // 2
                im0b[j00 : self.ny0 + j00, i00 : self.nx0 + i00] = im0
                im0 = im0b.reshape(
                    1,
                    1,
                    2 * self.displacement_max + self.ny1,
                    2 * self.displacement_max + self.nx1,
                )
            elif (self.ny0 > 2 * self.displacement_max + self.ny1) & (
                self.nx0 > 2 * self.displacement_max + self.nx1
            ):
                im0 = im0.reshape(1, 1, self.ny0, self.nx0)
        else:
            assert False, "Bad value for self.mode"

        correl = self.correlf(im0, im1)
        correl = np.asarray(correl)
        if (
            (self.ny0 > 2 * self.displacement_max + self.ny1)
            & (self.nx0 > 2 * self.displacement_max + self.nx1)
            & (self.mode == "disp")
        ):
            i00 = (self.nx0 - self.nx1 + 1) // 2 - self.displacement_max
            j00 = (self.ny0 - self.ny1 + 1) // 2 - self.displacement_max
            correl = correl[
                0,
                0,
                0,
                j00 : j00 + 2 * self.displacement_max + 1,
                i00 : i00 + 2 * self.displacement_max + 1,
            ]
        else:
            correl = correl.reshape(self.ny, self.nx)

        return correl, norm


class CorrelFFTBase(CorrelBase):
    """Correlations using fft."""

    _tag = "fft.base"

    def _init2(self):

        if self.displacement_max is not None:
            where_large_displacement = np.zeros(self.im0_shape, dtype=bool)

            for indices, v in np.ndenumerate(where_large_displacement):
                dx, dy = self.compute_displacement_from_indices(*indices[::-1])
                displacement = np.sqrt(dx ** 2 + dy ** 2)
                if displacement > self.displacement_max:
                    where_large_displacement[indices] = True

            self.where_large_displacement = where_large_displacement

        self._check_im_shape(self.im0_shape, self.im1_shape)

    def _check_im_shape(self, im0_shape, im1_shape):
        if im1_shape is None:
            im1_shape = im0_shape

        if im0_shape != im1_shape:
            raise ValueError(
                "with this correlation method the input images "
                "have to have the same shape."
            )

    def compute_displacements_from_correl(self, correl, norm=1.0):

        """Compute the displacement (with subpix) from a correlation."""

        if self.displacement_max is not None:
            correl = correl.copy()
            correl[self.where_large_displacement] = np.nan

        return super().compute_displacements_from_correl(correl, norm=norm)


class CorrelFFTNumpy(CorrelFFTBase):
    """Correlations using numpy.fft."""

    _tag = "np.fft"

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2))
        corr = ifft2(fft2(im0).conj() * fft2(im1)).real
        correl = np.fft.fftshift(corr[::-1, ::-1])
        return correl, norm


class CorrelFFTW(CorrelFFTBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""

    FFTClass = FFTW2DReal2Complex
    _tag = "fftw"

    def _init2(self):
        CorrelFFTBase._init2(self)
        n0, n1 = self.im0_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2)) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1))
        correl = np.fft.fftshift(corr[::-1, ::-1])
        return correl, norm


class CorrelCuFFT(CorrelFFTBase):
    _tag = "cufft"
    """Correlations using fluidimage.fft.CUFFT2DReal2Complex"""
    FFTClass = CUFFT2DReal2Complex

    def _init2(self):
        CorrelFFTBase._init2(self)
        n0, n1 = self.im0_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2)) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1)).real * im0.size ** 2
        correl = np.fft.fftshift(corr[::-1, ::-1])
        return correl, norm


class CorrelSKCuFFT(CorrelFFTBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""

    FFTClass = SKCUFFT2DReal2Complex
    _tag = "skcufft"

    def _init2(self):
        CorrelFFTBase._init2(self)
        n0, n1 = self.im0_shape
        self.op = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2)) * im0.size
        op = self.op
        corr = op.ifft(op.fft(im0).conj() * op.fft(im1))
        correl = np.fft.fftshift(corr[::-1, ::-1])
        return correl, norm


correlation_classes = {
    v._tag: v
    for k, v in locals().items()
    if k.startswith("Correl") and not k.endswith("Base")
}
