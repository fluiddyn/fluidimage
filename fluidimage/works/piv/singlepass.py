"""Piv work and subworks
========================

.. todo::

   - as in UVmat: better patch "thin-plate spline" (?). Add variables as
     in UVmat (NbCenter, Coord_tps, SubRange, U_tps, V_tps)

   - detect and save multipeaks. Add variables:

     * deltaxs_2ndpeak {ivec: float32}
     * deltays_2ndpeak {ivec: float32}
     * correl_2ndpeak {ivec: float32}


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

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

from ...data_objects.piv import ArrayCouple, HeavyPIVResults
from ...calcul.correl import PIVError, correlation_classes
from .. import BaseWork

from ...calcul.interpolate.thin_plate_spline_subdom import \
    ThinPlateSplineSubdom

from ...calcul.interpolate.griddata import griddata


class InterpError(ValueError):
    pass


class BaseWorkPIV(BaseWork):
    """Base class for PIV.

    This class is meant to be subclassed, not instantiated directly.

    """
    @classmethod
    def create_default_params(cls):
        params = ParamContainer(tag='params')
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        pass

    def __init__(self, params=None):

        if params is None:
            params = self.__class__.create_default_params()
        else:
            if params.piv0.method_subpix == "2d_gaussian2" and \
               params.piv0.nsubpix is not None:
                raise ValueError(
                    "Subpixel method '2d_gaussian2' doesn't require nsubpix. "
                    "params.piv0.nsubpix has to be equal to None")
        self.params = params

        overlap = params.piv0.grid.overlap
        if overlap >= 1:
            raise ValueError(
                'params.piv0.grid.overlap has to be smaller than 1')
        self.overlap = overlap

        shape_crop_im0 = params.piv0.shape_crop_im0
        shape_crop_im1 = params.piv0.shape_crop_im1
        if shape_crop_im1 is None:
            shape_crop_im1 = shape_crop_im0
        self.shape_crop_im0 = shape_crop_im0
        self.shape_crop_im1 = shape_crop_im1

        if isinstance(shape_crop_im0, int):
            n_interrogation_window0 = (shape_crop_im0, shape_crop_im0)
        elif isinstance(shape_crop_im0, tuple) and len(shape_crop_im0) == 2:
            n_interrogation_window0 = shape_crop_im0
        else:
            raise NotImplementedError(
                'For now, shape_crop_im0 has to be one or two integer!')

        if isinstance(shape_crop_im1, int):
            n_interrogation_window1 = (shape_crop_im1, shape_crop_im1)
        elif isinstance(shape_crop_im1, tuple) and len(shape_crop_im1) == 2:
            n_interrogation_window1 = shape_crop_im1
        else:
            raise NotImplementedError(
                'For now, shape_crop_im1 has to be one or two integer!')

        if (n_interrogation_window1[0] > n_interrogation_window0[0] or
                n_interrogation_window1[1] > n_interrogation_window0[1]):
            raise NotImplementedError(
                'shape_crop_im1 must be inferior or equal to shape_crop_im0')

        niw0 = self.n_interrogation_window0 = n_interrogation_window0
        niw1 = self.n_interrogation_window1 = n_interrogation_window1

        for size in niw0 + niw1:
            if size % 2 == 1:
                raise NotImplementedError(
                    'All dimensions of the cropped windows must be even.')

        self.niw0o2 = (int(niw0[0]//2), int(niw0[1]//2))
        self.niw1o2 = (int(niw1[0]//2), int(niw1[1]//2))

        self._init_correl()

    def _init_correl(self):
        niw0 = self.n_interrogation_window0
        niw1 = self.n_interrogation_window1
        try:
            correl_cls = correlation_classes[self.params.piv0.method_correl]
        except KeyError:
            raise ValueError(
                'params.piv0.method_correl should be in ' +
                str(correlation_classes.keys()))

        self.correl = correl_cls(im0_shape=niw0, im1_shape=niw1,
                                 method_subpix=self.params.piv0.method_subpix,
                                 nsubpix=self.params.piv0.nsubpix)

    def _prepare_with_image(self, im0):
        """Initialize the object with an image.

        .. todo::

           Better ixvecs and iyvecs (starting from 0 and padding is silly).

        """
        self.imshape0 = len_y, len_x = im0.shape
        niw = self.n_interrogation_window0
        niwo2 = self.niw0o2

        stepy = niw[0] - int(np.round(self.overlap*niw[0]))
        stepx = niw[1] - int(np.round(self.overlap*niw[1]))
        assert stepy >= 1
        assert stepx >= 1

        ixvecs = np.arange(niwo2[1], len_x-niwo2[1], stepx, dtype=int)
        iyvecs = np.arange(niwo2[0], len_y-niwo2[0], stepy, dtype=int)
        self.ixvecs = ixvecs
        self.iyvecs = iyvecs
        iyvecs, ixvecs = np.meshgrid(iyvecs, ixvecs)

        self.ixvecs_grid = ixvecs.flatten()
        self.iyvecs_grid = iyvecs.flatten()

    def calcul(self, couple):
        if isinstance(couple, SerieOfArraysFromFiles):
            couple = ArrayCouple(serie=couple)

        if not isinstance(couple, ArrayCouple):
            raise ValueError

        im0, im1 = couple.get_arrays()
        if not hasattr(self, 'ixvecs_grid'):
            self._prepare_with_image(im0)

        deltaxs, deltays, xs, ys, correls_max, correls, errors = \
            self._loop_vectors(im0, im1)

        result = HeavyPIVResults(
            deltaxs, deltays, xs, ys, errors,
            correls_max=correls_max, correls=correls,
            couple=deepcopy(couple), params=self.params)

        return result

    def _pad_images(self, im0, im1):
        """Pad images with zeros.

        .. todo::

           Choose correctly the variable npad.

        """
        npad = self.npad = max(self.niw0o2)
        tmp = [(npad, npad), (npad, npad)]
        im0pad = np.pad(im0 - im0.min(), tmp, 'constant')
        im1pad = np.pad(im1 - im1.min(), tmp, 'constant')
        return im0pad, im1pad

    def calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):

        xs = self.ixvecs_grid
        ys = self.iyvecs_grid

        ixs0_pad = self.ixvecs_grid + self.npad
        iys0_pad = self.iyvecs_grid + self.npad
        ixs1_pad = ixs0_pad
        iys1_pad = iys0_pad

        return xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad

    def _loop_vectors(self, im0, im1,
                      deltaxs_approx=None, deltays_approx=None):

        im0pad, im1pad = self._pad_images(im0, im1)

        xs, ys, ixs0_pad, iys0_pad, ixs1_pad, iys1_pad = \
            self.calcul_indices_vec(deltaxs_approx=deltaxs_approx,
                                    deltays_approx=deltays_approx)

        nb_vec = len(xs)

        correls = [None]*nb_vec
        errors = {}
        deltaxs = np.empty(xs.shape, dtype='float32')
        deltays = np.empty_like(deltaxs)
        correls_max = np.empty_like(deltaxs)

        for ivec in range(nb_vec):

            ixvec0 = ixs0_pad[ivec]
            iyvec0 = iys0_pad[ivec]
            ixvec1 = ixs1_pad[ivec]
            iyvec1 = iys1_pad[ivec]

            im0crop = self._crop_im0(ixvec0, iyvec0, im0pad)
            im1crop = self._crop_im1(ixvec1, iyvec1, im1pad)

            if im0crop.shape != self.n_interrogation_window0 or \
               im1crop.shape != self.n_interrogation_window1:
                deltaxs[ivec] = np.nan
                deltays[ivec] = np.nan
                correls_max[ivec] = np.nan
                errors[ivec] = 'Bad im_crop shape.'
                continue

            # print(im0crop.shape, im1crop.shape)
            correl, coef_norm = self.correl(im0crop, im1crop)
            if self.index_pass == 0 and \
               self.params.piv0.coef_correl_no_displ is not None:
                correl[self.correl.inds0] *= \
                    self.params.piv0.coef_correl_no_displ

            correls[ivec] = correl
            try:
                deltax, deltay, correl_max = \
                    self.correl.compute_displacement_from_correl(
                        correl, coef_norm=coef_norm)
            except PIVError as e:
                errors[ivec] = e.explanation
                try:
                    deltax, deltay, correl_max = \
                        e.results_compute_displacement_from_correl
                except AttributeError:
                    deltax = np.nan
                    deltay = np.nan
                    correl_max = np.nan

            if np.isnan(deltax) or np.isnan(deltay):
                errors[ivec] = 'Problem compute_displacement_from_correl.'

            deltaxs[ivec] = deltax
            deltays[ivec] = deltay
            correls_max[ivec] = correl_max

        if deltaxs_approx is not None:
            deltaxs += deltaxs_approx
            deltays += deltays_approx

        return deltaxs, deltays, xs, ys, correls_max, correls, errors

    def _crop_im0(self, ixvec, iyvec, im):
        niwo2 = self.niw0o2
        subim = im[iyvec - niwo2[0]:iyvec + niwo2[0],
                   ixvec - niwo2[1]:ixvec + niwo2[1]]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def _crop_im1(self, ixvec, iyvec, im):
        niwo2 = self.niw1o2
        subim = im[iyvec - niwo2[0]:iyvec + niwo2[0],
                   ixvec - niwo2[1]:ixvec + niwo2[1]]
        subim = np.array(subim, dtype=np.float32)
        return subim - subim.mean()

    def apply_interp(self, piv_results, last=False):
        couple = piv_results.couple

        im0, im1 = couple.get_arrays()
        if not last and not hasattr(piv_results, 'ixvecs_approx'):
            piv_results.ixvecs_approx = self.ixvecs_grid
            piv_results.iyvecs_approx = self.iyvecs_grid

        if last and not hasattr(piv_results, 'ixvecs_final'):
            piv_results.ixvecs_final = self.ixvecs_grid
            piv_results.iyvecs_final = self.iyvecs_grid

        # for the interpolation
        selection = ~(np.isnan(piv_results.deltaxs) |
                      np.isnan(piv_results.deltays))

        if not any(selection):
            raise InterpError('Only nan.')

        xs = piv_results.xs[selection]
        ys = piv_results.ys[selection]
        centers = np.vstack([ys, xs])

        deltaxs = piv_results.deltaxs[selection]
        deltays = piv_results.deltays[selection]

        if self.params.multipass.use_tps is True or \
           self.params.multipass.use_tps == 'last' and last:
            print('TPS interpolation.')
            # compute TPS coef
            smoothing_coef = self.params.multipass.smoothing_coef
            subdom_size = self.params.multipass.subdom_size
            threshold = self.params.multipass.threshold_tps

            tps = ThinPlateSplineSubdom(
                centers, subdom_size, smoothing_coef,
                threshold=threshold, pourc_buffer_area=0.5)
            try:
                deltaxs_smooth, deltaxs_tps = \
                    tps.compute_tps_coeff_subdom(deltaxs)
                deltays_smooth, deltays_tps = \
                    tps.compute_tps_coeff_subdom(deltays)
            except np.linalg.LinAlgError:
                print('compute delta_approx with griddata (in tps)')
                deltaxs_approx = griddata(centers, deltaxs,
                                          (self.iyvecs, self.ixvecs))
                deltays_approx = griddata(centers, deltays,
                                          (self.iyvecs, self.ixvecs))
            else:
                piv_results.deltaxs_smooth = deltaxs_smooth
                piv_results.deltaxs_tps = deltaxs_tps
                piv_results.deltays_smooth = deltays_smooth
                piv_results.deltays_tps = deltays_tps

                new_positions = np.vstack([
                    self.iyvecs_grid, self.ixvecs_grid])

                tps.init_with_new_positions(new_positions)

                deltaxs_approx = tps.compute_eval(deltaxs_tps)
                deltays_approx = tps.compute_eval(deltays_tps)
        else:
            deltaxs_approx = griddata(centers, deltaxs,
                                      (self.iyvecs, self.ixvecs))
            deltays_approx = griddata(centers, deltays,
                                      (self.iyvecs, self.ixvecs))

        if last:
            piv_results.deltaxs_final = deltaxs_approx
            piv_results.deltays_final = deltays_approx
        else:
            piv_results.deltaxs_approx = deltaxs_approx
            piv_results.deltays_approx = deltays_approx


class FirstWorkPIV(BaseWorkPIV):
    """Basic PIV pass."""
    index_pass = 0

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child('piv0', attribs={
            'shape_crop_im0': 48,
            'shape_crop_im1': None,
            'delta_max': None,
            'delta_mean': None,
            'method_correl': 'fftw',
            'method_subpix': '2d_gaussian',
            'nsubpix': 1,
            'coef_correl_no_displ': None})

        params.piv0._set_doc("""
Parameters describing one PIV step.

shape_crop_im0 : int (48)
    Shape of the cropped images 0 from which are computed the correlation.
shape_crop_im1 : int or None
    Shape of the cropped images 0 (has to be None for correl based on fft).
delta_max : None
    Displacement maximum.
delta_mean : None
    Displacement averaged over space.

method_correl : str, {'fftw', ...}

method_subpix : str, {'2d_gaussian', ...}

nsubpix : 1
    Integer used in the subpix finder. It is related to the typical size of the
    particles. It has to be increased in case of peak locking (plot the
    histograms of the displacements).
""")

        params.piv0._set_child('grid', attribs={
            'overlap': 0.5,
            'from': 'overlap'})

        params.piv0.grid._set_doc("""
Parameters describing the grid.

overlap : float (0.5)
    Number smaller than 1 defining the overlap between interrogation windows.

from : str {'overlap'}
    Keyword for the method from which is computed the grid.
""")

        params._set_child('mask', attribs={})


class WorkPIVFromDisplacement(BaseWorkPIV):
    """Work PIV working from already computed displacement (for multipass)."""

    def __init__(self, params=None, index_pass=1, shape_crop_im0=None,
                 shape_crop_im1=None):

        if params is None:
            params = self.__class__.create_default_params()

        self.params = params
        self.index_pass = index_pass

        if shape_crop_im0 is None:
            shape_crop_im0 = params.piv0.shape_crop_im0
        if shape_crop_im1 is None:
            shape_crop_im1 = params.piv0.shape_crop_im1

        self.shape_crop_im0 = shape_crop_im0
        self.shape_crop_im1 = shape_crop_im1

        shape_crop_im0 = tuple(int(n) for n in shape_crop_im0)
        shape_crop_im1 = tuple(int(n) for n in shape_crop_im1)

        niw0 = self.n_interrogation_window0 = shape_crop_im0
        niw1 = self.n_interrogation_window1 = shape_crop_im1

        for size in niw0 + niw1:
            if size % 2 == 1:
                raise NotImplementedError(
                    'All dimensions of the cropped windows must be even.')

        self.niw0o2 = (int(niw0[0]//2), int(niw0[1]//2))
        self.niw1o2 = (int(niw1[0]//2), int(niw1[1]//2))

        self.overlap = params.piv0.grid.overlap

        self._init_correl()

    def calcul(self, piv_results):
        """Calcul the piv.

        .. todo::

           Use the derivatives of the velocity to distort the image 1.

        """
        if not isinstance(piv_results, HeavyPIVResults):
            raise ValueError

        couple = piv_results.couple

        im0, im1 = couple.get_arrays()
        if not hasattr(self, 'ixvecs_grid'):
            self._prepare_with_image(im0)

        self.apply_interp(piv_results)

        deltaxs_approx = piv_results.deltaxs_approx
        deltays_approx = piv_results.deltays_approx

        deltaxs_approx = np.round(deltaxs_approx).astype('int32')
        deltays_approx = np.round(deltays_approx).astype('int32')

        deltaxs, deltays, xs, ys, correls_max, correls, errors = \
            self._loop_vectors(im0, im1,
                               deltaxs_approx=deltaxs_approx,
                               deltays_approx=deltays_approx)

        result = HeavyPIVResults(
            deltaxs, deltays, xs, ys, errors,
            correls_max=correls_max, correls=correls,
            couple=deepcopy(couple), params=self.params)

        return result

    def calcul_indices_vec(self, deltaxs_approx=None, deltays_approx=None):
        """Calcul the indices corresponding to the windows in im0 and im1.

        .. todo::

           Better handle calculus of indices for crop image center on
           image 0 and image 1.

        """
        ixs0 = self.ixvecs_grid - deltaxs_approx // 2
        iys0 = self.iyvecs_grid - deltays_approx // 2
        ixs1 = ixs0 + deltaxs_approx
        iys1 = iys0 + deltays_approx

        # if a point is outside an image => shift of subimages used
        # for correlation
        ind_outside = np.argwhere(
            (ixs0 > self.imshape0[1]) | (ixs0 < 0) |
            (ixs1 > self.imshape0[1]) | (ixs1 < 0) |
            (iys0 > self.imshape0[0]) | (iys0 < 0) |
            (iys1 > self.imshape0[0]) | (iys1 < 0))

        for ind in ind_outside:
            if ((ixs1[ind] > self.imshape0[1]) or
                    (iys1[ind] > self.imshape0[0]) or
                    (ixs1[ind] < 0) or (iys1[ind] < 0)):
                ixs0[ind] = self.ixvecs_grid[ind] - deltaxs_approx[ind]
                iys0[ind] = self.iyvecs_grid[ind] - deltays_approx[ind]
                ixs1[ind] = self.ixvecs_grid[ind]
                iys1[ind] = self.iyvecs_grid[ind]
            elif ((ixs0[ind] > self.imshape0[1]) or
                  (iys0[ind] > self.imshape0[0]) or
                  (ixs0[ind] < 0) or (iys0[ind] < 0)):
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
