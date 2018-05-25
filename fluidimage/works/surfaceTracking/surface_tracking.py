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


import datetime
import math

import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.interpolate
import scipy.io


from .. import BaseWork
from ...data_objects.surfaceTracking import SurfaceTrackingObject


class WorkSurfaceTracking(BaseWork):
    """Base class for SurfaceTracking

    ? This class is meant to be subclassed, not instantiated directly.

    """

    @classmethod
    def _complete_params_with_default(cls, params):
        params._set_child(
            "surface_tracking_params",
            attribs={
                "xmin": 475,  # 25  # x axis pixel range to crop the image imx[min:max]
                "xmax": 640,  # 275  # x axis pixel range to crop the image imx[min:max]
                "ymin": 50,  # y axis pixel range to crop the image imy[min:max]
                "ymax": 700,  # y axis pixel range to crop the image imy[min:max]
                "distance_lens": 0.36,  # distance in [m] lenses of camera/projetor
                "distance_object": 1.07,  # distance in [m] camera/projector and surface
                "pix_size": 2.4 * 10 ** -4,
                "startref_frame": 0,
                "lastref_frame": 49,
                "sur": 16,
                "k_x": 70.75,  # wave vector oj. grid (approx. value, will set accurate later)
                "k_y": 0,  # wave vector of the grid y-axis
                "slicer": 4,
                "bo": 1,  # cut the borders
                "red_factor": 1,  # reduction factor to for the pixels to take tp speed up
                "n_frames_stock": 1,  # number of frames to stock in one file
            },
        )
        pass

    def __init__(self, params):

        self.params = params

        self.works_surface_tracking = []
        self.nameFrame = None

        self.path = params.film.path
        self.pathRef = params.film.pathRef

        self.verify_process = False
        self.ref_film = None
        self.filmName = None
        self.save_png = True
        self.treshold = 0.16

        self.xmin = self.params.surface_tracking_params.xmin
        self.xmax = self.params.surface_tracking_params.xmax
        self.ymin = self.params.surface_tracking_params.ymin
        self.ymax = self.params.surface_tracking_params.ymax

        self.distance_lens = self.params.surface_tracking_params.distance_lens
        self.distance_object = self.params.surface_tracking_params.distance_object
        self.pix_size = self.params.surface_tracking_params.pix_size

        self.startref_frame = self.params.surface_tracking_params.startref_frame
        self.lastref_frame = self.params.surface_tracking_params.lastref_frame
        self.sur = self.params.surface_tracking_params.sur
        self.k_x = self.params.surface_tracking_params.k_x
        self.k_y = self.params.surface_tracking_params.k_y
        self.slicer = self.params.surface_tracking_params.slicer

        self.bo = self.params.surface_tracking_params.bo
        self.red_factor = self.params.surface_tracking_params.red_factor
        self.n_frames_stock = self.params.surface_tracking_params.n_frames_stock

        self.plot_reduction_factor = 10
        self.l_x = self.xmax - self.xmin
        self.l_y = self.ymax - self.ymin

        self.wave_proj = 1 / (self.k_x / self.l_x / self.pix_size)
        wave_proj_pix = self.wave_proj / self.pix_size
        self.kslicer = 2 * self.k_x

        self.kx = np.arange(-self.l_x / 2, self.l_x / 2) / self.l_x
        self.ky = np.arange(-self.l_y / 2, self.l_y / 2) / self.l_y

        self.kxx = self.kx / self.pix_size
        self.gain, self.filt = self.create_gainfilter(
            self.ky, self.kx, self.k_y, self.k_x, self.l_y, self.l_x, self.slicer
        )

    def compute(self, frame):

        surfaceTracking = SurfaceTrackingObject(params=self.params)

        H_sav, H_filt, status, ttlast = self.processAFrame(
            self.path,
            self.l_x,
            self.l_y,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.gain,
            self.filt,
            self.bo,
            self.red_factor,
            self.pix_size,
            self.distance_object,
            self.distance_lens,
            self.wave_proj,
            self.n_frames_stock,
            self.plot_reduction_factor,
            self.kx,
            self.ky,
            frame[0],
            verify_process=False,
            offset=2.5,
        )

        surfaceTracking.H_sav = H_sav
        surfaceTracking.H_filt = H_filt
        surfaceTracking.nameFrame = frame[1].split("/")[-1]
        return surfaceTracking
        # offset = thickness/2 #of the reference plate (in order to find the origin)

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def create_gainfilter(self, ky, kx, k_y, k_x, l_y, l_x, slicer):
        kxgrid, kygrid = np.meshgrid(kx, ky)
        X, Y = np.meshgrid(kx * l_x, ky * l_y)
        gain = np.exp(-1.j * 2 * np.pi * (k_x / l_x * X + k_y / l_y * Y))
        filt1 = np.fft.fftshift(
            np.exp(-((kxgrid ** 2 + kygrid ** 2) / 2 / (k_x / slicer / l_x) ** 2))
            * np.exp(
                1
                - 1 / (1 + ((kxgrid + k_x) ** 2 + (kygrid + k_y) ** 2) / k_x ** 2)
            )
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
        # filt4 = np.fft.fftshift(-np.exp(-(((kxgrid+k_x))**2 +
        #                        kygrid**2)/2/(k_x)**2)+1)
        # filt5 = np.fft.fftshift(-np.exp(-(((kxgrid+2*k_x/3))**2 +
        #                        kygrid**2)/2/(2*k_x/3)**2)+1)
        return gain, filt1 * filt2 * filt3

    def filters(self, frame, gain, filt):
        return np.fft.fft2(frame * gain) * filt

    def frame_adimx(self, frame):
        """ """
        meanx_frame = np.mean(frame, axis=1)
        for zz in range(np.shape(frame)[1]):
            frame[:, zz] = frame[:, zz] / meanx_frame
        frame = frame - np.mean(frame)
        return frame

    def process_frame(
        self, frame, ymin, ymax, xmin, xmax, gain, filt, bo, red_factor
    ):
        frame1 = frame[ymin:ymax, xmin:xmax].astype(float)
        frame1 = self.frame_adimx(frame1)
        Fref_filtre = self.filters(frame1, gain, filt)
        iFref_filtre = np.fft.ifft2(Fref_filtre)
        iFref_filtre = iFref_filtre[bo:-bo:red_factor, bo:-bo:red_factor]
        aref = np.unwrap(np.angle(iFref_filtre), axis=1)  # by lines
        aref = np.unwrap(aref, axis=0)  # by colums
        return aref

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
                frame1 = self.frame_adimx(frame1)
                ref = ref + frame1
                ref = ref / (
                    lastref_frame + 1 - startref_frame
                )  # STRANGE... I think it is not good... BONAMY
            ii += 1
        Fref = np.fft.fft2(ref, ((ymax - ymin) * sur, (xmax - xmin) * sur))
        kxma = np.arange(-(xmax - xmin) * sur / 2, (xmax - xmin) * sur / 2) / sur
        # kyma = np.arange(-l_y*sur/2, l_y*sur/2)/sur
        indc = np.max(np.fft.fftshift(abs(Fref)), axis=0).argmax()
        return ref, abs(kxma[indc])

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
        bo,
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
        verify_process=True,
        filmName=None,
        offset=0,
    ):

        start = datetime.datetime.now()
        status = False
        # initialize array H_sav and H_filt
        Ly = int((l_y - bo - bo - 1) / red_factor)  # length(list)
        Lx = int((l_x - bo - bo - 1) / red_factor)  # length(list)
        H_sav = np.zeros(
            (Ly - 8 + 1, Lx - 8 + 1, n_frames_stock), dtype=np.float32
        )
        H_filt = np.zeros(
            (Ly - 8 + 1, Lx - 8 + 1, n_frames_stock), dtype=np.float32
        )

        fix_y = int(np.fix(l_y / 2 / red_factor))
        fix_x = int(np.fix(l_x / 2 / red_factor))

        iframe = 1
        a2 = self.process_frame(
            frame, ymin, ymax, xmin, xmax, gain, filt, bo, red_factor
        )
        if iframe == 1:
            a_mem = a2
            jump = a2[fix_y, fix_x] - a_mem[fix_y, fix_x]
            while abs(jump) > math.pi:
                a2 = a2 - np.sign(jump) * 2 * math.pi
                jump = a2[fix_y, fix_x] - a_mem[fix_y, fix_x]
            a_mem = a2
        self.H = self.convphase(
            a2,
            pix_size,
            distance_object,
            distance_lens,
            self.wave_proj,
            "True",
            red_factor,
        )
        self.H = self.H[4:-4, 4:-4]
        Hfilt = self.H - np.mean(self.H)
        afilt = a2 - np.mean(a2) - offset

        #        if (save_png):
        #            #heights_plotter(-a2, f/plot_reduction_factor, 1, pix_size*1000, pix_size*1000)
        #            self.heights_plotter(-afilt, f/plot_reduction_factor, 1, pix_size*1000, pix_size*1000)

        #        if (iframe % n_frames_stock > 0):
        #            H_sav[:, :, iframe % n_frames_stock-1] = self.H
        #            H_filt[:, :, iframe % n_frames_stock-1] = Hfilt
        #        else:
        #         print("save as matlab hdf5-file; iframe from ", iframe-n_frames_stock,
        #               " to ", iframe-1)
        #         H_sav[:, :, n_frames_stock-1] = self.H
        #         H_filt[:, :, n_frames_stock-1] = Hfilt
        #         pathsav = path+"/results/"

        #            scipy.io.savemat(pathsav+str(f)+"heights_{:03d}".format(int(iframe/n_frames_stock)),
        #                             mdict={'H_sav': H_sav, 'H_filt': H_filt})
        if verify_process:
            plt.figure(3)
            plt.ion()
            pylab.imshow(self.H)
            plt.pause(0.05)

            plt.figure(4)
            plt.ion()
            pylab.imshow(Hfilt)
            plt.pause(0.05)

        if verify_process:
            X, Y = np.meshgrid(kx, ky)
            fig = plt.figure(5)
            ax = fig.add_subplot(211, projection="3d")
            # Plot the surface.
            self.surf = ax.plot_surface(
                X[0 : self.H.shape[0], 0 : self.H.shape[1]],
                Y[0 : self.H.shape[0], 0 : self.H.shape[1]],
                self.H_sav[:, :, 0],
                linewidth=0,
                antialiased=False,
            )
            ax = fig.add_subplot(212, projection="3d")
            # Plot the surface.
            self.surf = ax.plot_surface(
                X[0 : self.H.shape[0], 0 : self.H.shape[1]],
                Y[0 : self.H.shape[0], 0 : self.H.shape[1]],
                self.H_sav[:, :, 900],
                linewidth=0,
                antialiased=False,
            )
            plt.show()

        end_f = datetime.datetime.now()
        ttlast = (end_f - start).seconds + float(
            (end_f - start).microseconds
        ) / 1000000

        print("process film finished after ", ttlast, " s")
        status = True
        return H_sav, H_filt, status, ttlast

    def _prepare_with_image(self, im=None, nameFrame=None, imshape=None):
        """Prepare the works surface_tracking with an image."""
        if imshape is None:
            imshape = im.shape
        for work_surface_tracking in self.works_surface_tracking:
            work_surface_tracking._prepare_with_image(imshape=imshape)
