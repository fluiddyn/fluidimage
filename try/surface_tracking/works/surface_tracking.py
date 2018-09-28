"""Surface tracking (:mod:`fluidimage.works.surface_tracking`)
==============================================================

.. autofunction:: process_frame

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
import numpy as np
import scipy.interpolate
import scipy.io
import pims


from .. import BaseWork
from fluidimage.data_objects.old.surface_tracking import SurfaceTrackingObject


class WorkSurfaceTracking(BaseWork):
    """Base class for surface tracking

    """

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
                "distance_object": 1.07,
                "pix_size": 2.4 * 10 ** -4,
                "startref_frame": 0,
                "lastref_frame": 49,
                "sur": 16,
                "k_x": 70.75,
                "k_y": 0,
                "slicer": 4,
                "red_factor": 1,
                "n_frames_stock": 1,
            },
        )
        params.surface_tracking._set_doc(
            """
- xmin: 475,

  x axis pixel range to crop the image imx[min:max]

- xmax: 640,

  x axis pixel range to crop the image imx[min:max]

- ymin: 50,

  y axis pixel range to crop the image imy[min:max]

- ymax: 700,

  y axis pixel range to crop the image imy[min:max]

- distance_lens: 0.36,

  distance in [m] lenses of camera/projetor

- distance_object: 1.07,

  distance in [m] camera/projector and surface

- pix_size: 2.4 * 10 ** -4,

- startref_frame: 0,

- lastref_frame: 49,

- sur: 16,

- k_x: 70.75,

  wave vector oj. grid (approx. value, will set accurate later)

- k_y: 0,

  wave vector of the grid y-axis

- slicer: 4,

  cut the borders

- red_factor: 1,

  reduction factor to for the pixels to take tp speed up

- n_frames_stock: 1,

  number of frames to stock in one file
"""
        )

    def __init__(self, params):

        self.cpt = 0

        self.params = params

        self.works_surface_tracking = []
        self.nameFrame = None

        self.path = params.film.path
        self.path_ref = params.film.path_ref

        self.verify_process = False
        self.ref_film = None
        self.filmName = None
        self.save_png = True
        self.treshold = 0.16

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
        self.k_x = self.params.surface_tracking.k_x
        self.k_y = self.params.surface_tracking.k_y
        self.slicer = self.params.surface_tracking.slicer

        self.red_factor = self.params.surface_tracking.red_factor
        self.n_frames_stock = self.params.surface_tracking.n_frames_stock

        self.plot_reduction_factor = 10
        self.l_x = self.xmax - self.xmin
        self.l_y = self.ymax - self.ymin

        self.wave_proj = 1 / (self.k_x / self.l_x / self.pix_size)
        # wave_proj_pix = self.wave_proj / self.pix_size
        self.kslicer = 2 * self.k_x

        self.kx = np.arange(-self.l_x / 2, self.l_x / 2) / self.l_x
        self.ky = np.arange(-self.l_y / 2, self.l_y / 2) / self.l_y

        # self.refraw = self.get_file(self.path_ref)
        # self.refraw = self.sum_frame_from(self.path_ref)
        # refc, k_x = self.wave_vector(
        #    self.refraw,
        #    self.ymin,
        #    self.ymax,
        #    self.xmin,
        #    self.xmax,
        #    self.sur,
        #    self.startref_frame,
        #    self.lastref_frame,
        # )
        k_x = 70.75

        self.kxx = self.kx / self.pix_size
        self.gain, self.filt = self.set_gain_filter(
            k_x, self.l_y, self.l_x, self.slicer
        )
        self.a1_tmp = None

    def compute(self, frameCouple):
        """
        Compute a frame
        :param frameCouple: A couple of frame
        :type data_objects.piv.ArrayCouple
        :return:
        """
        surface_tracking = SurfaceTrackingObject(params=self.params)
        H, H_filt = self.processAFrame(
            self.path,
            self.l_x,
            self.l_y,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.gain,
            self.filt,
            self.red_factor,
            self.pix_size,
            self.distance_object,
            self.distance_lens,
            self.wave_proj,
            self.n_frames_stock,
            self.plot_reduction_factor,
            self.kx,
            self.ky,
            frameCouple,
            verify_process=False,
        )
        surface_tracking.H = H
        surface_tracking.H_filt = H_filt
        surface_tracking.pix_size = self.pix_size
        surface_tracking.nameFrame = frameCouple.arrays[0][1].split("/")[-1]
        return surface_tracking
        # of the reference plate (in order to find the origin)

    def processAFrame(
        self,
        path,
        l_x,
        l_y,
        xmin,
        xmax,
        ymin,
        ymax,
        gain,
        filt,
        red_factor,
        pix_size,
        distance_object,
        distance_lens,
        wave_proj,
        n_frames_stock,
        plot_reduction_factor,
        kx,
        ky,
        frame,
        save_png=True,
        verify_process=False,
        filmName=None,
    ):
        arrays1 = frame.arrays[0][0]
        arrays2 = frame.arrays[1][0]

        fix_y = int(np.fix(l_y / 2 / red_factor))
        fix_x = int(np.fix(l_x / 2 / red_factor))
        if self.a1_tmp is None:
            a1 = self.process_frame(
                arrays1, ymin, ymax, xmin, xmax, gain, filt, red_factor
            )
        else:
            a1 = self.a1_tmp
        a2 = self.process_frame(
            arrays2, ymin, ymax, xmin, xmax, gain, filt, red_factor
        )
        self.a1_tmp = a2

        jump = a2[fix_y, fix_x] - a1[fix_y, fix_x]

        while abs(jump) > math.pi:
            a2 = a2 - np.sign(jump) * 2 * math.pi
            jump = a2[fix_y, fix_x] - a1[fix_y, fix_x]

        H = self.convphase(
            a2,
            pix_size,
            distance_object,
            distance_lens,
            self.wave_proj,
            "True",
            red_factor,
        )
        # this line removes flickering from the projector
        Hfilt = H - np.mean(H)

        return H, Hfilt

    def set_gain_filter(self, k_x, l_y, l_x, slicer):
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

    def rectify_frame(self, frame, gain, filt):
        return np.fft.fft2(frame * gain) * filt

    def frame_normalize(self, frame):
        """normalize the frame values by its mean value"""
        meanx_frame = np.mean(frame, axis=1)
        for y in range(np.shape(frame)[1]):
            frame[:, y] = frame[:, y] / meanx_frame
        normalized_frame = frame - np.mean(frame)
        return normalized_frame

    def process_frame(
        self, frame, ymin, ymax, xmin, xmax, gain, filt, red_factor
    ):
        frame1 = frame[ymin:ymax, xmin:xmax]
        frame1 = self.frame_normalize(frame1).astype(float)
        frame_filtered = self.rectify_frame(frame1, gain, filt)
        inversed_filt = np.fft.ifft2(frame_filtered)
        inversed_filt = inversed_filt[::red_factor, ::red_factor]
        a = np.unwrap(np.angle(inversed_filt), axis=1)  # by lines
        a = np.unwrap(a, axis=0)  # by colums
        return a

    def process_frame1(self, im):
        return self.process_frame(
            im,
            self.ymin,
            self.ymax,
            self.xmin,
            self.xmax,
            self.gain,
            self.filt,
            self.red_factor,
        )

    def calculheight(self, ph):
        return self.convphase(
            ph,
            self.pix_size,
            self.distance_object,
            self.distance_lens,
            self.wave_proj,
            "True",
            self.red_factor,
        )

    def convphase(self, ph, pix_size, l, d, p, correct_pos, red_factor):
        """converts phase into height [m]
        make sure that the grid is parallel to y
        ph is the image phase [radians]
        pix_size  [m/pixel]
        l distance between object and camera [m]
        d distance between projector and camera [m]
        p wave length of the object [m]
        correct_pos if set 1 the position will be corrected
        red is the reduction factor"""

        height = ph * l / (ph - 2 * np.pi / p * d)
        if correct_pos is True:
            height = height.astype(float)
            [ld, Ld] = ph.shape()
            x = (range(Ld) - Ld / 2) * pix_size * red_factor
            y = (range(ld) - ld / 2) * pix_size * red_factor
            [X, Y] = np.meshgrid(x, y)
            # perform correction
            dX = -X / l * self.H
            dY = -Y / l * self.H
            dX[1, :] = 0
            dX[-1, :] = 0
            dX[:, 1] = 0
            dX[-1, :] = 0
            dY[1, :] = 0
            dY[-1, :] = 0
            dY[:, 1] = 0
            dY[-1, :] = 0
            # interploate the values on the new grid
            height = scipy.interpolate.griddata(
                X + dX, Y + dY, height, X, Y, "cubic"
            )
        return height

    def wave_vector(
        self, ref_film, ymin, ymax, xmin, xmax, sur, startref_frame, lastref_frame
    ):
        ref = np.zeros((ymax - ymin, xmax - xmin))
        ii = 0
        for frame in ref_film:
            if ii < lastref_frame - startref_frame:
                frame1 = frame[ymin:ymax, xmin:xmax].astype(float)
                frame1 = self.frame_normalize(frame1)
                ref = ref + frame1
            ii += 1
        ref = ref / (lastref_frame + 1 - startref_frame)
        Fref = np.fft.fft2(ref, ((ymax - ymin) * sur, (xmax - xmin) * sur))
        kxma = np.arange(-(xmax - xmin) * sur / 2, (xmax - xmin) * sur / 2) / sur
        # kyma = np.arange(-l_y*sur/2, l_y*sur/2)/sur
        indc = np.max(np.fft.fftshift(abs(Fref)), axis=0).argmax()
        return ref, abs(kxma[indc])

    def sum_frame_from(self, path="./", fn=None):
        """read the frames from file in 
        path or from a specified file if fn-arg is given
        and make the mean of the frame"""
        film = self.get_file(self.path)
        path = self.path
        if fn is None:
            # print(path + '/*')
            film = self.get_file(path, "film.cine")
            # film = pims.open(path + '/*')
        else:
            print("####" + path + "/" + fn)
            film = pims.open(str(path) + "/" + fn)
        return frame

    def get_file(self, path="./", fn=None):
        """read the files in path or a specified file if fn-arg is given"""
        path = self.path
        if fn is None:
            # print(path + '/*')
            film = self.get_file(path, "film.cine")
            # film = pims.open(path + '/*')
        else:
            print("####" + path + "/" + fn)
            film = pims.open(str(path) + "/" + fn)
        return film
