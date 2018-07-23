#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:04:32 2018

@author: blancfat8p
"""

import numpy as np
import math
import scipy

import h5netcdf
import h5py
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import scipy.io
import scipy.interpolate


from matplotlib import cm

mycmap = cm.get_cmap("viridis")
mycmap.set_under("w")

from .. import __version__ as fluidimage_version
from .._hg_rev import hg_rev


class DataObject(object):
    pass


class SurfaceTrackingObject(SerieOfArraysFromFiles):
    """
        Define objects for surface tracking
    """

    _keys_to_be_saved = ["H_sav", "H_filt", "status", "ttlast"]

    def __init__(self, params, file_name=None, str_path=None):

        self.passes = []
        self.params = params
        if file_name is not None:
            self.file_name = file_name
        if str_path is not None:
            self._load(str_path)
            return

        self.path_save_plot = (
            params.film.path + "." + params.saving.postfix + "/plot/"
        )
        self.path_save = (
            params.film.path + "." + params.saving.postfix + "/results/"
        )
        self.file_name = params.film.fileName

        # Matrix results
        self.H = None
        self.H_filt = None
        self.a_filt = None
        self.pix_size = None

        # hdf5 file
        self.f = None
        self.dset = None

        # plot
        self.mem_surface_tracking = []
        self.surf = None

    def save(self, nameFrame=None):
        """
        Save surface tracking results and call generate_plot
        :param nameFrame: Name of the frame
        :type str
        :return:
        """
        nameFrame = nameFrame.split("/")[-1]
        if self.params.saving.plot == True:
            self.generate_plot(
                -self.a_filt,
                nameFrame,
                1,
                self.pix_size * 1000,
                self.pix_size * 1000,
            )

        if self.params.saving.how == "mat":
            scipy.io.savemat(
                self.path_save + nameFrame,
                mdict={"H_sav": self.H, "H_filt": self.H_filt},
            )
        else:
            self.save_hdf5(nameFrame)

    def save_hdf5(self, nameFrame, path=None):
        """
        Save a result in HD5F in self.path_file
        :param nameFrame: Name of the frame
        :type str
        :param path: Saving path if different from self.path_file
        :type str
        :return:
        """
        # Managing how many results to put in a single hdf5 file
        numFrame = int(nameFrame.split("[")[1].split("]")[0])  # get the index
        suffix_save = str(
            (math.floor(numFrame / self.params.saving.how_many)) + 1
        )
        # Managing path
        if path is not None:
            path_file = path
        else:
            path_file = self.path_save

        with h5py.File(
            path_file + self.file_name + "_" + suffix_save + ".hdf5", "a"
        ) as f:
            f.attrs["class_name"] = "SurfaceTrackingObject"
            f.attrs["module_name"] = "fluidimage.data_objects.surface_tracking"
            f.attrs["fluidimage_version"] = fluidimage_version
            f.attrs["fluidimage_hg_rev"] = hg_rev
            try:
                f.create_dataset(nameFrame, data=self.H)
            except:
                pass
        return path_file

    def generate_plot(self, h, name, scale_h=1, scale_x=1, scale_y=1):
        """
        Generate and save a plot of surface tracking in self.path_save_plot
        :param h: a matrix representing surface tracking
        :type array
        :param name: Name of the string
        :type str
        :param scale_h:
        :param scale_x:
        :param scale_y:
        :return:
        """
        plt.ioff()
        y, x = np.meshgrid(np.arange(0, h.shape[1]), np.arange(0, h.shape[0]))

        h_s = h * scale_h
        x_s = x * scale_x
        y_s = y * scale_y
        # print(x_s.shape[1])
        # print(len(x_s))

        fig = plt.figure(figsize=(10, 6))

        ax1 = plt.subplot2grid(
            (5, 7), (0, 0), colspan=4, rowspan=3, projection="3d"
        )

        surf = ax1.plot_surface(
            x_s, y_s, h_s[:, :], cmap=mycmap, vmin=-20, vmax=20, antialiased=False
        )
        ax1.plot3D(
            x_s[:, -5],
            y_s[:, -5],
            h_s[:, -5],
            "--",
            color="darkblue",
            linewidth=2,
        )
        ax1.plot3D(
            x_s[-220, :],
            y_s[-220, :],
            h_s[-220, :],
            ":",
            color="darkblue",
            linewidth=1.5,
        )
        ax1.plot3D([0, 0], [0, 0], [-20, 20], "None")
        ax1.set_zlabel("Height [mm]")
        ax1.set_xlabel("Foil lateral [mm]")
        ax1.set_ylabel("Foil longitudal [mm]")
        fig.colorbar(
            surf, shrink=0.6, aspect=40, orientation="horizontal", pad=0.15
        )
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.view_init(30, -30)
        ax1.set_aspect(1.5)

        ax2 = plt.subplot2grid((5, 7), (3, 1), colspan=6, rowspan=2)
        plt.plot([-1, 150], [0, 0], "k", linewidth=0.5)
        #    plt.plot(np.arange(0, h.shape[0]), h_s[:, int(y.shape[1]*2/3)], '--',
        #             color='darkblue', linewidth=2)
        plt.plot(x_s[:, -120], h_s[:, -120], "--", color="darkblue", linewidth=2)
        ax2.set_ylim(-20, 20)
        ax2.set_xlabel("Cut lateral  [mm]")
        ax2.set_ylabel("Height [mm]")
        ax2.spines["right"].set_bounds(-25, 25)
        ax2.spines["left"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.yaxis.set_ticks_position("right")
        ax2.xaxis.set_ticks_position("bottom")
        ax2.yaxis.set_label_position("right")
        plt.grid(which="major", linestyle="-")
        plt.minorticks_on()
        ax2.set_yticks([-20, -10, 0, 10, 20])
        ax2.set_aspect("equal")
        ax3 = plt.subplot2grid((5, 7), (0, 4), colspan=3, rowspan=3)
        plt.plot([-1, 65], [0, 0], "k", linewidth=0.5)
        plt.plot(y_s[-220, :], h_s[-220, :], ":", color="darkblue", linewidth=2.5)
        # plt.plot(np.arange(0, h.shape[1]), h_s[int(x.shape[1]-12), :], ':',
        #         color='darkblue', linewidth=1.5)
        #  -np.mean(h_s[int(x.shape[1]-12), :]),
        ax3.set_ylim(-5, 5)
        # ax3.set_xticks(np.arange(0, 130, 25))
        ax3.set_xlabel("Cut longitudal [mm]")
        #  ax3.set_ylabel('Deformation to mean [mm]')
        ax3.set_ylabel("Height [mm]")
        ax3.spines["right"].set_bounds(-20, 20)
        ax3.spines["left"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.yaxis.set_ticks_position("right")
        ax3.xaxis.set_ticks_position("bottom")
        ax3.yaxis.set_label_position("right")
        plt.grid(which="major", linestyle="-")
        plt.minorticks_on()
        ax3.set_yticks([-20, -10, 0, 10, 20])
        ax3.set_aspect("equal")
        plt.tight_layout()
        png_file = self.path_save_plot + name + ".png"
        # plt.savefig(svg_file)
        plt.savefig(png_file)
        plt.close()
