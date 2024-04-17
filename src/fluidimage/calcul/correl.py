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

from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import correlate
from scipy.signal import correlate2d
from transonic import Array, Type, boost

from .correl_pycuda import correl_pycuda
from .errors import PIVError
from .fft import CUFFT2DReal2Complex, FFTW2DReal2Complex, SKCUFFT2DReal2Complex
from .subpix import SubPix


def compute_indices_from_displacement(dx, dy, indices_no_displ):
    return indices_no_displ[1] - dx, indices_no_displ[0] - dy


def parse_displacement_max(displ_max, im0_shape):
    if isinstance(displ_max, str) and displ_max.endswith("%"):
        return float(displ_max[:-1]) / 100 * max(im0_shape)

    else:
        return displ_max


A2dC = Array[Type(np.float32, np.float64), "2d", "C"]
A2df32 = "float32[][]"


def _is_there_a_nan(arr):
    arr = arr.ravel()
    for idx in range(9):
        if np.isnan(arr[idx]):
            return True
    return False


@boost
def nan_indices_max(
    correl: A2dC,
    i0_start: np.int32,
    i0_stop: np.int32,
    i1_start: np.int32,
    i1_stop: np.int32,
):

    correl_max = np.nan

    # first, get the first non nan value
    n0, n1 = correl.shape
    correl_flatten = correl.ravel()
    for i_flat in range(i0_start * n1 + i1_start, n0 * n1):
        value = correl_flatten[i_flat]
        if not np.isnan(value):
            correl_max = value
            break

    assert not np.isnan(correl_max)

    i0_max = 0
    i1_max = 0

    for i0 in range(i0_start, i0_stop):
        for i1 in range(i1_start, i1_stop):
            value = correl[i0, i1]
            if np.isnan(value):
                continue
            if value >= correl_max:
                correl_max = value
                i0_max = i0
                i1_max = i1

    error_message = ""

    i0, i1 = i0_max, i1_max

    if i0 == 0 or i0 == n0 - 1 or i1 == 0 or i1 == n1 - 1:
        error_message = "Correlation peak touching boundary."
    elif _is_there_a_nan(correl[i0 - 1 : i0 + 2, i1 - 1 : i1 + 2]):
        error_message = "Correlation peak touching nan."

    return i0_max, i1_max, error_message


def _compute_indices_max(
    correl, norm, start_stop_for_search0, start_stop_for_search1
):
    """Compute the indices of the maximum correlation

    Warning: important for perf, so use Pythran

    """
    i0_start, i0_stop = start_stop_for_search0
    i1_start, i1_stop = start_stop_for_search1

    if i0_stop is None:
        i0_stop, i1_stop = correl.shape

    iy, ix, error_message = nan_indices_max(
        correl, i0_start, i0_stop, i1_start, i1_stop
    )

    if norm == 0:
        # I hope it is ok (Pierre)
        correl_max = 0.0
    else:
        correl_max = correl[iy, ix] / norm

    if error_message:
        error = PIVError(explanation=error_message)
        error.results = (ix, iy, correl_max)
        raise error

    return ix, iy, correl_max


class CorrelBase(ABC):
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

        self.start_stop_for_search0 = [0, None]
        self.start_stop_for_search1 = [0, None]
        self._finalize_init()

    @abstractmethod
    def __call__(self, im0, im1):
        """Compute the correlation from 2 images."""

    def _finalize_init(self):
        """Finalize initialization"""

    def compute_displacement_from_indices(self, ix, iy):
        """Compute the displacement from a couple of indices."""
        return self.ix0 - ix, self.iy0 - iy

    def compute_indices_from_displacement(self, dx, dy):
        """Compute the indices corresponding to a displacement"""
        return self.ix0 - dx, self.iy0 - dy

    def get_indices_no_displacement(self):
        """Get the indices corresponding to no displacement"""
        return self.iy0, self.ix0

    def _compute_indices_max(self, correl, norm):
        return _compute_indices_max(
            correl, norm, self.start_stop_for_search0, self.start_stop_for_search1
        )

    def compute_displacements_from_correl(self, correl, norm=1.0):
        """Compute the displacement from a correlation."""

        try:
            ix, iy, correl_max = self._compute_indices_max(correl, norm)
        except PIVError as piv_error:
            ix, iy, correl_max = piv_error.results
            # second chance to find a better peak...
            correl[
                iy - self.particle_radius : iy + self.particle_radius + 1,
                ix - self.particle_radius : ix + self.particle_radius + 1,
            ] = np.nan
            try:
                ix2, iy2, correl_max2 = self._compute_indices_max(correl, norm)
            except PIVError as _piv_error:
                dx, dy = self.compute_displacement_from_indices(ix, iy)
                _piv_error.results = (dx, dy, correl_max)
                raise _piv_error

            else:
                ix, iy, correl_max = ix2, iy2, correl_max2

        dx, dy = self.compute_displacement_from_indices(ix, iy)

        if self.nb_peaks_to_search == 1:
            other_peaks = None
        elif self.nb_peaks_to_search >= 1:
            correl = correl.copy()
            other_peaks = []
            for _ in range(0, self.nb_peaks_to_search - 1):
                correl[
                    iy - self.particle_radius : iy + self.particle_radius + 1,
                    ix - self.particle_radius : ix + self.particle_radius + 1,
                ] = np.nan
                try:
                    ix, iy, correl_max_other = self._compute_indices_max(
                        correl, norm
                    )
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


@boost
def correl_numpy(im0: A2df32, im1: A2df32, disp_max: int):
    """Correlations by hand using only numpy.

    Parameters
    ----------

    im0, im1 : images
      input images : 2D matrix

    disp_max : int
      displacement max.

    Notes
    -----

    im1_shape inf to im0_shape

    Returns
    -------

    the computing correlation (size of computed correlation = disp_max*2 + 1)

    """
    norm = np.sqrt(np.sum(im1**2) * np.sum(im0**2))

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
    """Correlation computed "by hands" with Numpy and Pythran"""

    _tag = "pythran"

    def _finalize_init(self):
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
        return correl_numpy(im0, im1, self.displacement_max)


class CorrelPyCuda(CorrelBase):
    """Correlation using pycuda.
    Correlation class by hands with cuda.
    """

    _tag = "pycuda"

    def _finalize_init(self):
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
        # ???
        # self._add_info_to_correl(correl)
        return correl, norm


class CorrelScipySignal(CorrelBase):
    """Correlations using scipy.signal.correlate2d"""

    _tag = "scipy.signal"

    def _finalize_init(self):
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
        norm = _norm_images(im0, im1)
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

    def _finalize_init(self):
        self.iy0, self.ix0 = (i // 2 for i in self.im0_shape)

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = np.sum(im1**2)
        correl = correlate(im0, im1, mode="constant", cval=im1.min())
        return correl, norm


class CorrelFFTBase(CorrelBase):
    """Correlations using fft."""

    _tag = "fft.base"

    def _finalize_init(self):
        if self.displacement_max is not None:
            where_large_displacement = np.zeros(self.im0_shape, dtype=bool)

            for indices, v in np.ndenumerate(where_large_displacement):
                dx, dy = self.compute_displacement_from_indices(*indices[::-1])
                displacement = np.sqrt(dx**2 + dy**2)
                if displacement > self.displacement_max:
                    where_large_displacement[indices] = True

            self.where_large_displacement = where_large_displacement

            n0, n1 = where_large_displacement.shape
            for i0_start in range(n0):
                if not all(where_large_displacement[i0_start, :]):
                    break
            for i1_start in range(n1):
                if not all(where_large_displacement[:, i1_start]):
                    break
            for i0_stop in range(n0 - 1, -1, -1):
                if not all(where_large_displacement[i0_stop, :]):
                    break
            i0_stop += 1
            for i1_stop in range(n1 - 1, -1, -1):
                if not all(where_large_displacement[:, i1_stop]):
                    break
            i1_stop += 1

            self.start_stop_for_search0 = (i0_start, i0_stop)
            self.start_stop_for_search1 = (i1_start, i1_stop)

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


@boost
def _norm_images(im0: A2df32, im1: A2df32):
    """Less accurate than the numpy equivalent but much faster

    Should return something close to:

    np.sqrt(np.sum(im1**2) * np.sum(im0**2))

    """
    im0 = im0.ravel()
    im1 = im1.ravel()
    tmp0 = np.float64(im0[0] ** 2)
    tmp1 = np.float64(im1[0] ** 2)
    if im0.size != im1.size:
        for idx in range(1, im0.size):
            tmp0 += im0[idx] ** 2
        for idx in range(1, im1.size):
            tmp1 += im1[idx] ** 2
    else:
        for idx in range(1, im0.size):
            tmp0 += im0[idx] ** 2
            tmp1 += im1[idx] ** 2
    return np.sqrt(tmp0 * tmp1)


@boost
def _like_fftshift(arr: A2dC):
    """Pythran optimized function doing the equivalent of

    np.ascontiguousarray(np.fft.fftshift(arr[::-1, ::-1]))

    """
    n0, n1 = arr.shape

    assert n0 % 2 == 0
    assert n1 % 2 == 0

    arr = np.ascontiguousarray(arr[::-1, ::-1])
    tmp = np.empty_like(arr)

    for i0 in range(n0):
        for i1 in range(n1 // 2):
            tmp[i0, n1 // 2 + i1] = arr[i0, i1]
            tmp[i0, i1] = arr[i0, n1 // 2 + i1]

    arr_1d_view = arr.ravel()
    tmp_1d_view = tmp.ravel()

    n_half = n0 * n1 // 2
    for idx in range(n_half):
        arr_1d_view[idx + n_half] = tmp_1d_view[idx]
        arr_1d_view[idx] = tmp_1d_view[idx + n_half]

    return arr


class CorrelFFTNumpy(CorrelFFTBase):
    """Correlations using numpy.fft."""

    _tag = "np.fft"

    def __call__(self, im0, im1):
        """Compute the correlation from images."""
        norm = _norm_images(im0, im1)
        correl = ifft2(fft2(im0).conj() * fft2(im1)).real
        return _like_fftshift(np.ascontiguousarray(correl)), norm


class CorrelFFTWithOperBase(CorrelFFTBase):

    FFTClass: object

    def _finalize_init(self):
        CorrelFFTBase._finalize_init(self)
        n0, n1 = self.im0_shape
        self.oper = self.FFTClass(n1, n0)

    def __call__(self, im0, im1):
        """Compute the correlation from images.

        Warning: important for perf, so use Pythran

        """
        norm = _norm_images(im0, im1) * im0.size
        oper = self.oper
        correl = oper.ifft(oper.fft(im0).conj() * oper.fft(im1))
        return _like_fftshift(correl), norm


class CorrelFFTW(CorrelFFTWithOperBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""

    FFTClass = FFTW2DReal2Complex
    _tag = "fftw"


class CorrelCuFFT(CorrelFFTWithOperBase):
    """Correlations using fluidimage.fft.CUFFT2DReal2Complex"""

    _tag = "cufft"
    FFTClass = CUFFT2DReal2Complex


class CorrelSKCuFFT(CorrelFFTWithOperBase):
    """Correlations using fluidimage.fft.FFTW2DReal2Complex"""

    FFTClass = SKCUFFT2DReal2Complex
    _tag = "skcufft"


correlation_classes = {
    v._tag: v
    for k, v in locals().items()
    if k.startswith("Correl") and not k.endswith("Base")
}
