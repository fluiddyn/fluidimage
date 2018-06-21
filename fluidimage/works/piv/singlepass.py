"""Piv work and subworks
========================

.. todo::

   - fix using secondary peak in correlation

   - displacement_max

   - displacement_mean


.. autoclass:: BaseWorkPIV
   :members:
   :private-members:

.. autoclass:: FirstWorkPIV
   :members:
   :private-members:

.. autoclass:: WorkPIVFromDisplacement
   :members:
   :private-members:

"""

from __future__ import print_function

from copy import deepcopy

import numpy as np

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ...data_objects.piv import (
    ArrayCouple,
    HeavyPIVResults,
    get_slices_from_strcrop,
)
from ...calcul.correl import correlation_classes
from .. import BaseWork

from ...calcul.interpolate.thin_plate_spline_subdom import ThinPlateSplineSubdom

from ...calcul.interpolate.griddata import griddata
from ...calcul.subpix import SubPix
from ...calcul.errors import PIVError


class InterpError(ValueError):
    pass


def _isint(obj):
    return isinstance(obj, (int, np.integer))


class BaseWorkPIV(BaseWork):
    """Base class for PIV.

    This class is meant to be subclassed, not instantiated directly.

    """

    @classmethod
    def _complete_params_with_default(cls, params):
        pass

    def _init_shape_crop(self, shape_crop_im0, shape_crop_im1):

        if shape_crop_im1 is None:
            shape_crop_im1 = shape_crop_im0

        if _isint(shape_crop_im0):
            shape_crop_im0 = (shape_crop_im0, shape_crop_im0)
        elif (
            isinstance(shape_crop_im0, (tuple, list)) and len(shape_crop_im0) == 2
        ):
            shape_crop_im0 = tuple(shape_crop_im0)
        else:
            raise NotImplementedError(
                "For now, shape_crop_im0 has to be one or two integer!"
            )

        if _isint(shape_crop_im1):
            shape_crop_im1 = (shape_crop_im1, shape_crop_im1)
        elif (
            isinstance(shape_crop_im1, (tuple, list)) and len(shape_crop_im1) == 2
        ):
            shape_crop_im1 = tuple(shape_crop_im1)
        else:
            raise NotImplementedError(
                "For now, shape_crop_im1 has to be one or two integer!"
            )

        if (
            shape_crop_im1[0] > shape_crop_im0[0]
            or shape_crop_im1[1] > shape_crop_im0[1]
        ):
            raise ValueError(
                "shape_crop_im1 must be inferior or equal to shape_crop_im0"
            )

        self.shape_crop_im0 = tuple(int(n) for n in shape_crop_im0)
        self.shape_crop_im1 = tuple(int(n) for n in shape_crop_im1)

    def _init_correl(self):
        try:
            correl_cls = correlation_classes[self.params.piv0.method_correl]
        except KeyError:
            raise ValueError(
                "params.piv0.method_correl should be in "
                + str(list(correlation_classes.keys()))
            )

        self.correl = correl_cls(
            im0_shape=self.shape_crop_im0,
            im1_shape=self.shape_crop_im1,
            method_subpix=self.params.piv0.method_subpix,
            nsubpix=self.params.piv0.nsubpix,
            displacement_max=self.params.piv0.displacement_max,
            particle_radius=self.params.piv0.particle_radius,
            nb_peaks_to_search=self.params.piv0.nb_peaks_to_search,
        )

    def _prepare_with_image(self, im0=None, imshape=None):
        """Initialize the object with an image.
        """
        if imshape is None:
            imshape = im0.shape

        self.imshape0 = (len_y, len_x) = imshape
        scim = self.shape_crop_im0

        stepy = scim[0] - int(np.round(self.overlap * scim[0]))
        stepx = scim[1] - int(np.round(self.overlap * scim[1]))
        assert stepy >= 1
        assert stepx >= 1

        ixvec_max = len_x - self._stop_for_crop0[1]
        ixvecs = np.arange(self._start_for_crop0[1], ixvec_max, stepx, dtype=int)

        iyvec_max = len_y - self._stop_for_crop0[0]
        iyvecs = np.arange(self._start_for_crop0[0], iyvec_max, stepy, dtype=int)

        # There are some cases for which it is worth to add the last points...
        if ixvec_max - ixvecs[-1] > stepx // 2.5:
            # print('add another point (x)')
            ixvecs = np.append(ixvecs, ixvec_max)
        if iyvec_max - iyvecs[-1] > stepy // 2.5:
            # print('add another point (y)')
            iyvecs = np.append(iyvecs, iyvec_max)

        self.ixvecs = ixvecs
        self.iyvecs = iyvecs

        iyvecs, ixvecs = np.meshgrid(iyvecs, ixvecs)

        self.ixvecs_grid = ixvecs.flatten()
        self.iyvecs_grid = iyvecs.flatten()

    def calcul(self, couple):
        """Calcul the PIV (one pass) from a couple of images."""
        if isinstance(couple, SerieOfArraysFromFiles):
            couple = ArrayCouple(serie=couple)

        if not isinstance(couple, ArrayCouple):
            raise ValueError

        couple.apply_mask(self.params.mask)

        im0, im1 = couple.get_arrays()
        if not hasattr(self, "ixvecs_grid"):
            self._prepare_with_image(im0)
        (
            deltaxs,
            deltays,
            xs,
            ys,
            correls_max,
            correls,
            errors,
            secondary_peaks,
        ) = self._loop_vectors(im0, im1)

        xs, ys = self._xyoriginalimage_from_xymasked(xs, ys)

        result = HeavyPIVResults(
            deltaxs,
            deltays,
            xs,
            ys,
            errors,
            correls_max=correls_max,
            correls=correls,
            couple=deepcopy(couple),
            params=self.params,
            secondary_peaks=secondary_peaks,
        )

        self._complete_result(result)

        return result

    def _complete_result(self, result):
        result.indices_no_displacement = self.correl.get_indices_no_displacement()
        result.displacement_max = self.correl.displacement_max

    def _pad_images(self, im0, im1):
        """Pad images with zeros.

        .. todo::

           Choose correctly the variable npad.

        """
        npad = self.npad = max(self._start_for_crop0 + self._stop_for_crop0)
        tmp = [(npad, npad), (npad, npad)]
        im0pad = np.pad(im0 - im0.min(), tmp, "constant")
        im1pad = np.pad(im1 - im1.min(), tmp, "constant")
        return im0pad, im1pad

    def _calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):
        """Calcul the indices corresponding to the vectors and cropped windows.

        Returns
        -------

        xs : np.array

          x index of the position of the computed vector in the original
          images.

        ys : np.array

          y index of the position of the computed vector in the original
          images.

        ixs0_pad : np.array

          x index of the center of the crop image 0 in the padded image 0.

        iys0_pad : np.array

          y index of the center of the crop image 0 in the padded image 0.

        ixs1_pad : np.array

          x index of the center of the crop image 1 in the padded image 1.

        iys1_pad : np.array

          y index of the center of the crop image 1 in the padded image 1.

        """
        ixs0_pad = self.ixvecs_grid + self.npad
        iys0_pad = self.iyvecs_grid + self.npad
        ixs1_pad = ixs0_pad
        iys1_pad = iys0_pad

        return (
            self.ixvecs_grid,
            self.iyvecs_grid,
            ixs0_pad,
            iys0_pad,
            ixs1_pad,
            iys1_pad,
        )

    def _loop_vectors(self, im0, im1, deltaxs_approx=None, deltays_approx=None):
        """Loop over the vectors to compute them."""

        im0pad, im1pad = self._pad_images(im0, im1)

        xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad = self._calcul_indices_vec(
            deltaxs_approx=deltaxs_approx, deltays_approx=deltays_approx
        )

        nb_vec = len(xs)

        correls = [None] * nb_vec
        errors = {}
        deltaxs = np.empty(xs.shape, dtype="float32")
        deltays = np.empty_like(deltaxs)
        correls_max = np.empty_like(deltaxs)
        secondary_peaks = [None] * nb_vec

        has_to_apply_subpix = self.index_pass == self.params.multipass.number - 1

        for ivec in range(nb_vec):

            ixvec0 = ixs0_pad[ivec]
            iyvec0 = iys0_pad[ivec]
            ixvec1 = ixs1_pad[ivec]
            iyvec1 = iys1_pad[ivec]

            im0crop = self._crop_im0(ixvec0, iyvec0, im0pad)
            im1crop = self._crop_im1(ixvec1, iyvec1, im1pad)

            if (
                im0crop.shape != self.shape_crop_im0
                or im1crop.shape != self.shape_crop_im1
            ):

                print(
                    "Warning: Bad im_crop shape.",
                    ixvec0,
                    iyvec0,
                    ixvec1,
                    iyvec1,
                    im0crop.shape,
                    self.shape_crop_im0,
                    im1crop.shape,
                    self.shape_crop_im1,
                )

                deltaxs[ivec] = np.nan
                deltays[ivec] = np.nan
                correls_max[ivec] = np.nan
                errors[ivec] = "Bad im_crop shape."
                continue

            # compute and store correlation map
            correl, norm = self.correl(im0crop, im1crop)
            if (
                self.index_pass == 0
                and self.params.piv0.coef_correl_no_displ is not None
            ):
                correl[
                    self.correl.get_indices_no_displacement()
                ] *= self.params.piv0.coef_correl_no_displ

            correls[ivec] = correl

            # compute displacements corresponding to peaks
            try:
                deltax, deltay, correl_max, other_peaks = self.correl.compute_displacements_from_correl(
                    correl, norm=norm
                )
            except PIVError as e:
                errors[ivec] = e.explanation
                deltaxs[ivec], deltays[ivec], correls_max[ivec] = e.results
                continue

            # increase precision on the displacement
            if has_to_apply_subpix:
                try:
                    deltax, deltay = self.correl.apply_subpix(
                        deltax, deltay, correl
                    )
                except PIVError as e:
                    errors[ivec] = e.explanation

            deltaxs[ivec] = deltax
            deltays[ivec] = deltay
            correls_max[ivec] = correl_max

            secondary_peaks[ivec] = other_peaks

        if deltaxs_approx is not None:
            deltaxs += deltaxs_approx
            deltays += deltays_approx

        return (
            deltaxs,
            deltays,
            xs,
            ys,
            correls_max,
            correls,
            errors,
            secondary_peaks,
        )

    def _init_crop(self):
        """Initialize the cropping of the images."""

        scim0 = self.shape_crop_im0
        scim1 = self.shape_crop_im1

        self._start_for_crop0 = (scim0[0] // 2, scim0[1] // 2)
        _stop_for_crop0 = [scim0[0] // 2, scim0[1] // 2]

        if scim0[0] % 2 == 1:
            _stop_for_crop0[0] += 1

        if scim0[1] % 2 == 1:
            _stop_for_crop0[1] += 1

        self._stop_for_crop0 = tuple(_stop_for_crop0)

        self._start_for_crop1 = (scim1[0] // 2, scim1[1] // 2)
        _stop_for_crop1 = [scim1[0] // 2, scim1[1] // 2]

        if scim1[0] % 2 == 1:
            _stop_for_crop1[0] += 1

        if scim1[1] % 2 == 1:
            _stop_for_crop1[1] += 1

        self._stop_for_crop1 = tuple(_stop_for_crop1)

    def _crop_im0(self, ixvec, iyvec, im):
        """Crop image 0."""
        subim = im[
            iyvec - self._start_for_crop0[0] : iyvec + self._stop_for_crop0[0],
            ixvec - self._start_for_crop0[1] : ixvec + self._stop_for_crop0[1],
        ]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_im1(self, ixvec, iyvec, im):
        """Crop image 1."""
        subim = im[
            iyvec - self._start_for_crop1[0] : iyvec + self._stop_for_crop1[0],
            ixvec - self._start_for_crop1[1] : ixvec + self._stop_for_crop1[1],
        ]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _xymasked_from_xyoriginalimage(self, xs, ys):
        if self.params.mask.strcrop is not None:
            slices = get_slices_from_strcrop(self.params.mask.strcrop)
            if slices[1].start is not None:
                xs = xs - slices[1].start
            if slices[0].start is not None:
                ys = ys - slices[0].start
        return xs, ys

    def _xyoriginalimage_from_xymasked(self, xs, ys):
        if self.params.mask.strcrop is not None:
            slices = get_slices_from_strcrop(self.params.mask.strcrop)
            if slices[1].start is not None:
                xs = xs + slices[1].start
            if slices[0].start is not None:
                ys = ys + slices[0].start
        return xs, ys

    def apply_interp(self, piv_results, last=False):
        """Interpolate a PIV result object on the grid of the PIV work.

        Parameters
        ----------

        piv_results : :class;`HeavyPIVResults`

            The interpolated field are added to this object.

        last : bool (False)

            Last pass or not.

        Notes
        -----

        Depending on the value of `params.multipass.use_tps`, the interpolation
        is done with an interative method based on the Thin Plate Spline method
        (:class:`fluidimage.calcul.interpolate.thin_plate_spline_subdom.ThinPlateSplineSubdom`)
        or done with a simple griddata method (much faster).

        """

        if not last and not hasattr(piv_results, "ixvecs_approx"):
            piv_results.ixvecs_approx, piv_results.iyvecs_approx = self._xyoriginalimage_from_xymasked(
                self.ixvecs_grid, self.iyvecs_grid
            )

        if last and not hasattr(piv_results, "ixvecs_final"):
            piv_results.ixvecs_final, piv_results.iyvecs_final = self._xyoriginalimage_from_xymasked(
                self.ixvecs_grid, self.iyvecs_grid
            )

        # for the interpolation
        selection = ~(
            np.isnan(piv_results.deltaxs) | np.isnan(piv_results.deltays)
        )

        if not any(selection):
            raise InterpError("Only nan.")

        xs = piv_results.xs[selection]
        ys = piv_results.ys[selection]

        xs, ys = self._xymasked_from_xyoriginalimage(xs, ys)

        centers = np.vstack([ys, xs])

        deltaxs = piv_results.deltaxs[selection]
        deltays = piv_results.deltays[selection]

        if (
            self.params.multipass.use_tps is True
            or self.params.multipass.use_tps == "last"
            and last
        ):
            print("TPS interpolation ({}).".format(piv_results.couple.name))
            # compute TPS coef
            smoothing_coef = self.params.multipass.smoothing_coef
            subdom_size = self.params.multipass.subdom_size
            threshold = self.params.multipass.threshold_tps

            tps = ThinPlateSplineSubdom(
                centers,
                subdom_size,
                smoothing_coef,
                threshold=threshold,
                percent_buffer_area=0.25,
            )
            try:
                deltaxs_smooth, deltaxs_tps = tps.compute_tps_coeff_subdom(
                    deltaxs
                )
                deltays_smooth, deltays_tps = tps.compute_tps_coeff_subdom(
                    deltays
                )
            except np.linalg.LinAlgError:
                print("compute delta_approx with griddata (in tps)")
                deltaxs_approx = griddata(
                    centers, deltaxs, (self.iyvecs, self.ixvecs)
                )
                deltays_approx = griddata(
                    centers, deltays, (self.iyvecs, self.ixvecs)
                )
            else:
                piv_results.deltaxs_smooth = deltaxs_smooth
                piv_results.deltaxs_tps = deltaxs_tps
                piv_results.deltays_smooth = deltays_smooth
                piv_results.deltays_tps = deltays_tps

                new_positions = np.vstack([self.iyvecs_grid, self.ixvecs_grid])

                tps.init_with_new_positions(new_positions)

                deltaxs_approx = tps.compute_eval(deltaxs_tps)
                deltays_approx = tps.compute_eval(deltays_tps)
        else:
            deltaxs_approx = griddata(
                centers, deltaxs, (self.iyvecs, self.ixvecs)
            )
            deltays_approx = griddata(
                centers, deltays, (self.iyvecs, self.ixvecs)
            )

        if last:
            piv_results.deltaxs_final = deltaxs_approx
            piv_results.deltays_final = deltays_approx
        else:
            piv_results.deltaxs_approx = deltaxs_approx
            piv_results.deltays_approx = deltays_approx


class FirstWorkPIV(BaseWorkPIV):
    """First PIV pass (without input displacements).

    Parameters
    ----------

    params : ParamContainer

      ParamContainer object produced by the function
      :func:`fluidimage.works.piv.multipass.WorkPIV.create_default_params`.

    """

    index_pass = 0

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child(
            "piv0",
            attribs={
                "shape_crop_im0": 48,
                "shape_crop_im1": None,
                "displacement_max": None,
                "displacement_mean": None,  # NotImplemented
                "method_correl": "fftw",
                "method_subpix": "2d_gaussian2",
                "nsubpix": None,
                "coef_correl_no_displ": None,
                "nb_peaks_to_search": 1,
                "particle_radius": 3,
            },
        )

        params.piv0._set_doc(
            """Parameters describing one PIV step.

shape_crop_im0 : int (48)
    Shape of the cropped images 0 from which are computed the correlation.
shape_crop_im1 : int or None
    Shape of the cropped images 0 (has to be None for correl based on fft).

displacement_max : None
    Displacement maximum used in correlation classes. The exact effect depends
    on the correlation method. For fft based correlation, it can also be of the
    form '50%' and then the maximum displacement is computed for each pass as a
    pourcentage of max(shape_crop_im0).

displacement_mean : None
    Displacement averaged over space (NotImplemented).

method_correl : str, default 'fftw'

    Can be in """
            + str(list(correlation_classes.keys()))
            + """

method_subpix : str, default '2d_gaussian2'

    Can be in """
            + str(SubPix.methods)
            + """

nsubpix : None
    Integer used in the subpix finder. It is related to the typical size of the
    particles. It has to be increased in case of peak locking (plot the
    histograms of the displacements).

coef_correl_no_displ : None, number
    If this coefficient is not None, the correlation of the point corresponding
    to no displacement is multiplied by this coefficient (for the first pass).

nb_peaks_to_search : 1, int
    Number of peaks to search. Secondary peaks can be used during the fix step.

particle_radius : 3, int
    Typical radius of a particle (or more preciselly of a correlation
    peak). Used only if `nb_peaks_to_search` is larger than one.

"""
        )

        params.piv0._set_child(
            "grid", attribs={"overlap": 0.5, "from": "overlap"}
        )

        params.piv0.grid._set_doc(
            """
Parameters describing the grid.

overlap : float (0.5)
    Number smaller than 1 defining the overlap between interrogation windows.

from : str {'overlap'}
    Keyword for the method from which is computed the grid.
"""
        )

        params._set_child("mask", attribs={"strcrop": None})

        params.mask._set_doc(
            """
Parameters describing how images are masked.

strcrop : None, str

    Two-dimensional slice (for example '100:600, :'). If None, the whole image
    is used.
"""
        )

    def __init__(self, params):

        self.params = params

        self.overlap = params.piv0.grid.overlap
        if self.overlap >= 1:
            raise ValueError("params.piv0.grid.overlap has to be smaller than 1")

        self._init_shape_crop(
            params.piv0.shape_crop_im0, params.piv0.shape_crop_im1
        )
        self._init_crop()
        self._init_correl()


class WorkPIVFromDisplacement(BaseWorkPIV):
    """Work PIV working from already computed displacement (for multipass).

    Parameters
    ----------

    params : ParamContainer

      ParamContainer object produced by the function
      :func:`fluidimage.works.piv.multipass.WorkPIV.create_default_params`.

    index_pass : int, 1

    shape_crop_im0 : int or tuple, optional

    shape_crop_im1 : int or tuple, optional

    Notes
    -----

    Steps for the PIV computation:

    - prepare the work PIV with an image if not already done (grid),

    - apply interpolation (TPS or griddata) to compute a estimation of the
      displacements,

    - loop over vectors to compute displacements.

    """

    def __init__(
        self, params, index_pass=1, shape_crop_im0=None, shape_crop_im1=None
    ):

        self.params = params
        self.index_pass = index_pass

        self.overlap = params.piv0.grid.overlap

        if shape_crop_im0 is None:
            shape_crop_im0 = params.piv0.shape_crop_im0
        if shape_crop_im1 is None:
            shape_crop_im1 = params.piv0.shape_crop_im1

        self._init_shape_crop(shape_crop_im0, shape_crop_im1)
        self._init_crop()
        self._init_correl()

    def calcul(self, piv_results):
        """Calcul the PIV (one pass) from a couple of images and displacement.

        .. todo::

           Use the derivatives of the velocity to distort the image 1.

        """
        if not isinstance(piv_results, HeavyPIVResults):
            raise ValueError

        couple = piv_results.couple

        im0, im1 = couple.get_arrays()
        if not hasattr(self, "ixvecs_grid"):
            self._prepare_with_image(im0)

        self.apply_interp(piv_results)

        deltaxs_approx = piv_results.deltaxs_approx
        deltays_approx = piv_results.deltays_approx

        deltaxs_approx = np.round(deltaxs_approx).astype("int32")
        deltays_approx = np.round(deltays_approx).astype("int32")

        (
            deltaxs,
            deltays,
            xs,
            ys,
            correls_max,
            correls,
            errors,
            secondary_peaks,
        ) = self._loop_vectors(
            im0, im1, deltaxs_approx=deltaxs_approx, deltays_approx=deltays_approx
        )

        xs, ys = self._xyoriginalimage_from_xymasked(xs, ys)

        result = HeavyPIVResults(
            deltaxs,
            deltays,
            xs,
            ys,
            errors,
            correls_max=correls_max,
            correls=correls,
            couple=deepcopy(couple),
            params=self.params,
            secondary_peaks=secondary_peaks,
        )

        self._complete_result(result)
        result.deltaxs_approx0 = deltaxs_approx
        result.deltays_approx0 = deltays_approx

        return result

    def _calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):
        """Calcul the indices corresponding to the vectors and cropped windows.

        Returns
        -------

        xs : np.array

          x index of the position of the computed vector in the original
          images.

        ys : np.array

          y index of the position of the computed vector in the original
          images.

        ixs0_pad : np.array

          x index of the center of the crop image 0 in the padded image 0.

        iys0_pad : np.array

          y index of the center of the crop image 0 in the padded image 0.

        ixs1_pad : np.array

          x index of the center of the crop image 1 in the padded image 1.

        iys1_pad : np.array

          y index of the center of the crop image 1 in the padded image 1.

        """
        ixs0 = self.ixvecs_grid - deltaxs_approx // 2
        iys0 = self.iyvecs_grid - deltays_approx // 2
        ixs1 = ixs0 + deltaxs_approx
        iys1 = iys0 + deltays_approx

        # if a point is outside an image => shift of subimages used
        # for correlation
        ind_outside = np.argwhere(
            (ixs0 > self.imshape0[1])
            | (ixs0 < 0)
            | (ixs1 > self.imshape0[1])
            | (ixs1 < 0)
            | (iys0 > self.imshape0[0])
            | (iys0 < 0)
            | (iys1 > self.imshape0[0])
            | (iys1 < 0)
        )

        for ind in ind_outside:
            if (
                (ixs1[ind] > self.imshape0[1])
                or (iys1[ind] > self.imshape0[0])
                or (ixs1[ind] < 0)
                or (iys1[ind] < 0)
            ):
                ixs0[ind] = self.ixvecs_grid[ind] - deltaxs_approx[ind]
                iys0[ind] = self.iyvecs_grid[ind] - deltays_approx[ind]
                ixs1[ind] = self.ixvecs_grid[ind]
                iys1[ind] = self.iyvecs_grid[ind]
            elif (
                (ixs0[ind] > self.imshape0[1])
                or (iys0[ind] > self.imshape0[0])
                or (ixs0[ind] < 0)
                or (iys0[ind] < 0)
            ):
                ixs0[ind] = self.ixvecs_grid[ind]
                iys0[ind] = self.iyvecs_grid[ind]
                ixs1[ind] = self.ixvecs_grid[ind] + deltaxs_approx[ind]
                iys1[ind] = self.iyvecs_grid[ind] + deltays_approx[ind]

        xs = (ixs0 + ixs1) / 2.
        ys = (iys0 + iys1) / 2.

        ixs0_pad = ixs0 + self.npad
        iys0_pad = iys0 + self.npad
        ixs1_pad = ixs1 + self.npad
        iys1_pad = iys1 + self.npad

        return xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad
