"""Surface tracking (:mod:`fluidimage.works.surface_tracking`)
==============================================================

.. autoclass:: WorkSurfaceTracking
   :members:
   :private-members:

"""
###############################################################################
# !/usr/bin/env python                                                        #
#  -*- coding: utf-8 -*-                                                      #
#                         (C) Cyrille Bonamy, Stefan Hoerner, 2017            #
#            LEGI Grenoble, University Otto-von-Guericke Magdeburg            #
###############################################################################
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the GNU General Public License as published by        #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
# See the GNU General Public License for more details.                        #
# You should have received a copy of the GNU General Public License           #
# along with this program.                                                    #
# If not, see <http://www.gnu.org/licenses/>.                                 #
###############################################################################
#     This function provides surface tracking tools, it is part of the        #
#            oscillating profile experiment at LEGI 2017                      #
###############################################################################

import math
import sys
from pathlib import Path

import numpy as np
import scipy.interpolate
import scipy.io
from skimage.transform import resize

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage import SerieOfArraysFromFiles
from fluidimage.util import imread, logger

from . import BaseWork

# import matplotlib.pyplot as plt


class WorkSurfaceTracking(BaseWork):
    """Main work for surface tracking

    Parameters
    ----------

    params : :class:`fluiddyn.util.paramcontainer.ParamContainer`

      The default parameters are obtained from the class method
      :func:`WorkSurfaceTracking.create_default_params`.

    """

    @classmethod
    def create_default_params(cls):
        "Create an object containing the default parameters (class method)."
        params = ParamContainer(tag="params")
        cls._complete_params_with_default(params)
        return params

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child(
            "surface_tracking",
            attribs={
                "xmin": 475,
                "xmax": 640,
                "ymin": 50,
                "ymax": 700,
                "distance_lens": 0.36,
                "distance_object": 1.293,
                "pix_size": 2.4 * 10 ** -4,
                "startref_frame": 0,
                "lastref_frame": 49,
                "sur": 16,
                "k_x": 40.625,
                "k_y": 0,
                "slicer": 4,
                "red_factor": 1,
                "n_frames_stock": 1,
                "crop_edge": False,
                "borders": 1,
                "correct_pos": False,
                "correct_height": False,
                "offset": 0.0,
            },
        )

        params.surface_tracking._set_doc(
            """Surface Tracking parameters:

xmin: int (default 475)
    xmin to crop the image im[xmin:xmax, ymin:ymax].

xmax: int (default 640)
    xmax to crop the image im[xmin:xmax, ymin:ymax]

ymin: int (default 50)
    ymin to crop the image im[xmin:xmax, ymin:ymax]

ymax: int (default 700)
    ymax to crop the image im[xmin:xmax, ymin:ymax]

distance_lens: float (default 0.36)
    distance in [m] lenses of camera/projetor

distance_object: float (default 1.07)
    distance in [m] camera/projector and surface

pix_size: float (default 2.4 * 10 ** -4)
    pixel size

startref_frame: int (default 0)
    indice of first reference image

lastref_frame: int (default 49)
    indice of last reference image

sur: int (default 16)

k_x: float (default 70.75)
    wave vector oj. grid (approx. value, will set accurate later)

k_y: float (default 0)
    wave vector of the grid y-axis

slicer: int (default 4)
    cut the borders

red_factor: int (default 1)
    reduction factor to for the pixels to take tp speed up

n_frames_stock: int (default 1)
    number of frames to stock in one file

crop_edge: boolean (default False)
    searches for the structure and crops the part of the frame outside of
    the structure

borders: int (default 7)
    pixel to set zero height additional to the borders if the structure was
    cropped to avoid jerks

correct_pos: boolean (default=False)
    correct position of the height (necessary for large heights)

correct_height: boolean (default=False)
    correct height by a reference (provided from path_ref)

offset: float (default 0.0)
    height of the reference surface in [m] to zero level
"""
        )

    def __init__(self, params):

        self.cpt = 0

        self.params = params

        self.works_surface_tracking = []
        self.nameFrame = None

        self.path = params.images.path
        self.path_ref = params.images.path_ref

        self.verify_process = False
        self.ref_film = None
        self.filmName = None
        self.save_png = True
        self.thres = 300
        self.crop_edge = self.params.surface_tracking.crop_edge
        self.borders = self.params.surface_tracking.borders
        self.correct_pos = self.params.surface_tracking.correct_pos
        self.correct_height = self.params.surface_tracking.correct_height
        self.offset = self.params.surface_tracking.offset
        self.xmin = self.params.surface_tracking.xmin
        self.xmax = self.params.surface_tracking.xmax
        self.ymin = self.params.surface_tracking.ymin
        self.ymax = self.params.surface_tracking.ymax

        self.distance_lens = self.params.surface_tracking.distance_lens
        self.distance_object = self.params.surface_tracking.distance_object
        self.pix_size = self.params.surface_tracking.pix_size

        self.startref_frame = self.params.surface_tracking.startref_frame
        self.lastref_frame = self.params.surface_tracking.lastref_frame
        self.sur = self.params.surface_tracking.sur

        self.k_y = self.params.surface_tracking.k_y
        self.slicer = self.params.surface_tracking.slicer

        self.red_factor = self.params.surface_tracking.red_factor
        self.n_frames_stock = self.params.surface_tracking.n_frames_stock

        self.plot_reduction_factor = 10
        self.l_x = self.xmax - self.xmin
        self.l_y = self.ymax - self.ymin

        # wave_proj_pix = self.wave_proj / self.pix_size

        self.kx = np.arange(-self.l_x / 2, self.l_x / 2) / self.l_x
        self.ky = np.arange(-self.l_y / 2, self.l_y / 2) / self.l_y

        self.refserie = SerieOfArraysFromFiles(
            params.images.path_ref, params.images.str_slice_ref
        )
        k_x = self.compute_kx(self.refserie)
        logger.info("Value of kx computed = " + str(k_x))
        self.kslicer = 2 * k_x
        self.wave_proj = 1 / (k_x / self.l_x / self.pix_size)
        self.kxx = self.kx / self.pix_size
        self.gain, self.filt = self.set_gain_filter(
            k_x, self.l_y, self.l_x, self.slicer
        )
        self.a1_tmp = None
        self.ref_height = self.process_ref()
        logger.info("reference computed")

    def compute_kx(self, serie):
        """calculates the average wave vector from a set of reference images

        Parameters
        ----------
        series: int
        set of reference frames (arrays)

        Returns
        -------

        wave_vector: float
            average wave vector from the reference frame
        """
        if len(serie) == 0:
            logger.warning("0 ref image. Use of default k_x = 40.625.")
            return 40.625
        names = serie.get_path_arrays()
        ref = np.zeros((self.ymax - self.ymin, self.xmax - self.xmin))
        ii = 0

        for name in names:
            array = imread(str(Path(self.path_ref) / name))
            frame = array[self.ymin : self.ymax, self.xmin : self.xmax].astype(
                float
            )
            frame = self.frame_normalize(frame)
            ref = ref + frame
            ii += 1
        self.ref = ref / ii
        return self.wave_vector(
            self.ref, self.ymin, self.ymax, self.xmin, self.xmax, self.sur
        )

    def set_gain_filter(self, k_x, l_y, l_x, slicer):
        """compute gain and filter"""
        kx = np.arange(-l_x / 2, l_x / 2) / l_x
        ky = np.arange(-l_y / 2, l_y / 2) / l_y
        kxgrid, kygrid = np.meshgrid(kx, ky)
        X, Y = np.meshgrid(kx * l_x, ky * l_y)
        gain = np.exp(-1.0j * 2 * np.pi * (k_x / l_x * X))
        filt1 = np.fft.fftshift(
            np.exp(-((kxgrid ** 2 + kygrid ** 2) / 2 / (k_x / slicer / l_x) ** 2))
            * np.exp(1 - 1 / (1 + ((kxgrid + k_x) ** 2 + kygrid ** 2) / k_x ** 2))
        )

        filt2 = np.fft.fftshift(
            -np.exp(
                -(
                    ((kxgrid + (k_x / l_x)) ** 2 + kygrid ** 2)
                    / 2
                    / (k_x / 10 / l_x) ** 2
                )
            )
            + 1
        )
        filt3 = np.fft.fftshift(
            -np.exp(
                -(
                    ((kxgrid - (k_x / l_x)) ** 2 + kygrid ** 2)
                    / 2
                    / (k_x / 10 / l_x) ** 2
                )
            )
            + 1
        )
        return gain, filt1 * filt2 * filt3

    def get_borders(self, frame):
        """find the left and right border of the surface in a given frame

        Parameters
        ----------

        frame: int
            frame of the high speed video (np.array)

        Returns
        -------
        xmin: int
            left border of the structure
        xmax: int
            right border of the structure
        """
        frame_thres = 1.0 * (frame > self.thres)
        a = np.argmax(frame_thres, axis=1)
        xmin = np.median(a)
        b = np.argmax(frame_thres[:, ::-1], axis=1)
        b_med = np.median(b)
        xmax = frame.shape[1] - b_med
        return int(xmin), int(xmax)

    def merge_cropped_frame(self, frame, x_min, x_max):
        """puts the actual frame in the reference plate frame to avoid jerks
        and to keep the dimensions

        Parameters
        ----------

        frame: int
            frame of the high speed video (np.array)

        Returns
        -------

        calc_frame: int
            frame of the reference size with embedded smaller structure
        """
        if x_max >= self.xmax:
            x_max = self.xmax - 1
            print("INFO:x_max adjusted")
        if x_min <= self.xmin:
            x_min = self.xmin
            print("INFO:x_min adjusted")
        calc_frame = self.ref
        calc_frame[:, x_min - self.xmin : -(self.xmax - x_max)] = frame[
            self.ymin : self.ymax, x_min:x_max
        ]
        return calc_frame

    def rectify_frame(self, frame, gain, filt):
        """rectify a frame with gain and filt

        Parameters
        ----------

        frame: int
            frame of the high speed video (np.array)
        gain: complex array
            gain for the pattern of the frame

        filt: complex array
            filter for the pattern of the frame        gain:

        Returns
        -------

        rectified frame: int
            array of the rectified frame
        """
        return np.fft.fft2(frame * gain) * filt

    def frame_normalize(self, frame):
        """normalize the frame values by its mean value

        Parameters
        ----------

        frame: int
            frame of the high speed video (np.array)

        Returns
        -------
        normalized_frame: int
            normalized frame of the high speed video (np.array)
        """
        meanx_frame = np.mean(frame, axis=1)
        for y in range(np.shape(frame)[1]):
            frame[:, y] = frame[:, y] / meanx_frame
        normalized_frame = frame - np.mean(frame)
        return normalized_frame

    def process_frame(
        self, frame, ymin, ymax, xmin, xmax, gain, filt, red_factor
    ):
        """process a frame and return phase
        Parameters
        ----------

        frame: int array
            single frame of the high speed video

        xmin: int (default 475)
            xmin to crop the image im[xmin:xmax, ymin:ymax]

        xmax: int (default 640)
            xmax to crop the image im[xmin:xmax, ymin:ymax]

        ymin: int (default 50)
            ymin to crop the image im[xmin:xmax, ymin:ymax]

        ymax: int (default 700)
            ymax to crop the image im[xmin:xmax, ymin:ymax]

        gain: complex array
            gain for the pattern of the frame

        filt: complex array
            filter for the pattern of the frame

        red_factor: int(default 1)
            reduction factor for the frame array to speed up the calc

        Returns
        --------
            a: array containing phase [radians]

        """
        if self.crop_edge is False:
            frame = frame[ymin:ymax, xmin:xmax]
        frame1 = self.frame_normalize(frame).astype(float)
        frame_filtered = self.rectify_frame(frame1, gain, filt)
        inversed_filt = np.fft.ifft2(frame_filtered)
        inversed_filt = inversed_filt[::red_factor, ::red_factor]
        a = np.unwrap(np.angle(inversed_filt), axis=1)  # by lines
        a = np.unwrap(a, axis=0)  # by colums
        return a

    def process_frame_func(self, array_and_path):
        """call process_frame function with surface_tracking parameters

        Parameters
        ----------

        array_and_path : tuple containing array and path

        Returns
        -------

        array_and_path : tuple containing array/phase [radians], frame shape
        and path

        """
        array, path = array_and_path

        x_min = self.xmin
        x_max = self.xmax
        if self.crop_edge:
            x_min, x_max = self.get_borders(array)
            array = self.merge_cropped_frame(array, x_min, x_max)
        shape = (x_min, x_max)
        return (
            self.process_frame(
                array,
                self.ymin,
                self.ymax,
                self.xmin,
                self.xmax,
                self.gain,
                self.filt,
                self.red_factor,
            ),
            shape,
            path,
        )

    def calculheight_func(self, array_and_path):
        """call convphase function with surface_tracking parameters

        Parameters
        ----------

        array_and_path : tuple containing array/phase [radians],
        shape of the frame and path

        Returns
        -------

        height_and_path : tuple containing array/height [m], frame shape
        and path

        """
        array, shape, path = array_and_path
        array_ = []
        for a in array:
            jumps = [
                np.sign(int(a[i + 1] - angle))
                for i, angle in enumerate(a[:-1])
                if a[i + 1] - angle > 0.95 * np.pi
                or a[i + 1] - angle < 0.95 * np.pi
            ]

            mapper = np.zeros(len(jumps))
            smoother = []
            switch = 0

            for i, val in enumerate(mapper):
                if jumps[i] > 0:
                    switch = switch + 1
                if jumps[i] < 0:
                    switch = switch - 1
                if switch >= 1:
                    val = -2 * np.pi * switch
                if switch <= -1:
                    val = 2 * np.pi * switch
                smoother.append(val)
            smoother.insert(0, 0)
            array_smoothed = a + smoother
            array_.append(np.array(array_smoothed))
        array_s = np.array(array_)
        array_s = array_s.T
        for a in array_s:
            jumps = [
                np.sign(int(a[i + 1] - angle))
                for i, angle in enumerate(a[:-1])
                if a[i + 1] - angle > 0.95 * np.pi
                or a[i + 1] - angle < 0.95 * np.pi
            ]

            mapper = np.zeros(len(jumps))
            smoother = []
            switch = 0

            for i, val in enumerate(mapper):
                if jumps[i] > 0:
                    switch = switch + 1
                if jumps[i] < 0:
                    switch = switch - 1
                if switch >= 1:
                    val = -2 * np.pi
                if switch <= -1:
                    val = 2 * np.pi
                smoother.append(val)
            smoother.insert(0, 0)
            array_smoothed = a + smoother
            array_.append(np.array(array_smoothed))
        array_s = array
        return (
            self.convphase(
                array_s,
                self.pix_size,
                self.distance_object,
                self.distance_lens,
                self.wave_proj,
                self.red_factor,
            ),
            shape,
            path,
        )

    def set_borders_zero_func(self, array_and_path):
        """call convphase function with surface_tracking parameters

        Parameters
        ----------

        array_and_path : tuple containing array/phase [radians], shape of
        the frame  and path

        Returns
        -------

        height_and_path : tuple containing array/height [m] and path

        """
        array, shape, path = array_and_path
        (x_min, x_max) = shape
        if x_max >= self.xmax:
            x_max = self.xmax - 1
            logger.warning("x_max adjusted")
        newarray = np.zeros(self.ref.shape)
        newarray[:, self.borders : -self.borders] = resize(
            array[
                :,
                x_min
                + self.borders
                - self.xmin : -(self.xmax - x_max + self.borders),
            ],
            (self.ref.shape[0], self.xmax - 2 * self.borders - self.xmin),
        )
        return (newarray, path)

    def convphase(self, phase, pix_size, dist, dist_p_c, wave_len, red_factor):
        """converts phase array into array of the height [m]

        Parameters
        ----------

        phase : float
            the image phase [radians]

        pix_size  : float
            size of the pixel [m/pixel]

        dist : float
            distance between object and camera [m]

        dist_p_c : float
            distance between projector and camera [m]

        waven_len : float
            wave length of the object [m]

        red_factor: int
            is the reduction factor

        Returns
        -------

        height: float
            array with the height calculated from the phase

        Notes
        -----

        Make sure that the grid is parallel to y

        """

        height = phase * dist / (phase - 2 * np.pi / wave_len * dist_p_c)
        if self.correct_pos is True:
            height = height.astype(float)
            [ld, Ld] = phase.shape
            x = (np.arange(Ld) - Ld / 2) * pix_size * red_factor
            y = (np.arange(ld) - ld / 2) * pix_size * red_factor
            [X, Y] = np.meshgrid(x, y)
            # perform correction
            dX = -X / dist * height
            dY = -Y / dist * height
            dX[1, :] = 0
            dX[-1, :] = 0
            dX[:, 1] = 0
            dX[:-1] = 0
            dY[1, :] = 0
            dY[-1, :] = 0
            dY[:, 1] = 0
            dY[:-1] = 0
            # interploate the values on the new grid
            height = scipy.interpolate.griddata(
                ((X + dX).reshape(ld * Ld), (Y + dY).reshape(ld * Ld)),
                height.reshape(Ld * ld),
                (X, Y),
                method="cubic",
            )
        if self.correct_height is True:
            height = height - self.ref_height
        # height = height-np.mean(np.mean(height))
        return height

    def correctcouple(self, queue_couple):
        """correct phase in order to avoid jump phase"""
        (
            (anglemod, shapemod, path_anglemod),
            (angle, shape, path_angle),
        ) = queue_couple
        fix_y = int(np.fix(self.l_y / 20 / self.red_factor))
        fix_x = int(np.fix(self.l_x / 2 / self.red_factor))
        correct_angle = angle
        jump = angle[fix_y, fix_x] - anglemod[fix_y, fix_x]
        while abs(jump) > np.pi:
            correct_angle = correct_angle - np.sign(jump) * 2 * math.pi
            jump = correct_angle[fix_y, fix_x] - anglemod[fix_y, fix_x]
            print("angle corrected")
        return (correct_angle, shape, path_angle)

    def wave_vector(self, ref, ymin, ymax, xmin, xmax, sur):
        """compute k_x value with mean reference frame

        Parameters
        ----------

        ref: int
            frame with averaged

        Returns
        -------
        wave_vector: float
            average wave vector from the given frame
        """
        Fref = np.fft.fft2(ref, ((ymax - ymin) * sur, (xmax - xmin) * sur))
        kxma = np.arange(-(xmax - xmin) * sur / 2, (xmax - xmin) * sur / 2) / sur
        indc = np.max(np.fft.fftshift(abs(Fref)), axis=0).argmax()
        return abs(kxma[indc])

    def process_ref(self):
        """calculate the reference height from a set of frames of a flat
        plate with zero height
        """
        frame_filtered = self.rectify_frame(self.ref, self.gain, self.filt)
        inversed_filt = np.fft.ifft2(frame_filtered)
        inversed_filt = inversed_filt[:: self.red_factor, :: self.red_factor]
        ref_angle = np.unwrap(np.angle(inversed_filt), axis=1)  # by lines
        ref_angle = np.unwrap(ref_angle, axis=0)  # by columsref
        self.ref_height = np.zeros(ref_angle.shape)
        return (
            self.calculheight_func((ref_angle, ref_angle.shape, self.path_ref))[0]
            - self.offset
        )


if "sphinx" in sys.modules:
    params = WorkSurfaceTracking.create_default_params()
    __doc__ += params._get_formatted_docs()
