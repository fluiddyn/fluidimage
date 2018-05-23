#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:04:32 2018

@author: blancfat8p
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from matplotlib import cm


def get_str_index(serie, i, index):
    if serie._from_movies and i == serie.nb_indices - 1:
        return str(index)

    if serie._index_types[i] == "digit":
        code_format = "{:0" + str(serie._index_lens[i]) + "d}"
        str_index = code_format.format(index)
    elif serie._index_types[i] == "alpha":
        if index > 25:
            raise ValueError('"alpha" index larger than 25.')

        str_index = chr(ord("a") + index)
    else:
        raise Exception('The type should be "digit" or "alpha".')

    return str_index


def get_name_surface_tracking(frame, prefix="sf"):
    index_slices = frame._index_slices
    str_indices = ""
    for i, inds in enumerate(index_slices):
        index = inds[0]
        str_index = get_str_index(frame, i, index)
        if len(inds) > 1:
            str_index += "-" + get_str_index(frame, i, inds[1] - 1)

        if i > 1:
            str_indices += frame._index_separators[i - 1]
        str_indices += str_index

    name = prefix + "_" + str_indices + ".h5"
    return name




class DataObject(object):
    pass


class SurfaceTrackingObject(SerieOfArraysFromFiles):
    """
        Define objects for surface tracking
    """
    i = 0


    def __init__(self,params,file_name=None, str_path=None):
        SurfaceTrackingObject.i += 1
        self.params = params
        if file_name is not None:
            self.file_name = file_name
        if str_path is not None:
            self._load(str_path)
            return

        self.H_sav = None
        self.H_filt = None
        self.path_save = ""

    def generate_plot(self,h , name, scale_h=1, scale_x=1, scale_y=1):
        print("save")
        print(h)
        plt.ioff()
        y, x = np.meshgrid(np.arange(0, h.shape[1]), np.arange(0, h.shape[0]))
    
        h_s = h*scale_h
        x_s = x*scale_x
        y_s = y*scale_y
        # print(x_s.shape[1])
        # print(len(x_s))
    
        fig = plt.figure(figsize=(10, 6))
    
        ax1 = plt.subplot2grid((5, 7), (0, 0),
                               colspan=4, rowspan=3, projection='3d')
        
        self.surf = ax1.plot_surface(x_s, y_s, h_s[:, :], cmap=cm.viridis,
                                vmin=-20, vmax=20,antialiased=False)
        ax1.plot3D(x_s[:,-5], y_s[:,-5],
                   h_s[:, -5], '--',
                   color='darkblue', linewidth=2)
        ax1.plot3D(x_s[-220,:], y_s[-220,:],
                   h_s[-220,:], ':',
                   color='darkblue', linewidth=1.5)
        ax1.plot3D([0, 0], [0, 0], [-20, 20], 'None')
        ax1.set_zlabel('Height [mm]')
        ax1.set_xlabel('Foil lateral [mm]')
        ax1.set_ylabel('Foil longitudal [mm]')
        fig.colorbar(self.surf, shrink=0.6, aspect=40, orientation="horizontal",
                     pad=0.15)
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.view_init(30, -30)
        ax1.set_aspect(1.5)
    
        ax2 = plt.subplot2grid((5, 7), (3, 1), colspan=6, rowspan=2)
        plt.plot([-1, 150], [0, 0], 'k', linewidth=0.5)
    #    plt.plot(np.arange(0, h.shape[0]), h_s[:, int(y.shape[1]*2/3)], '--',
    #             color='darkblue', linewidth=2)
        plt.plot(x_s[:, -120], h_s[:, -120], '--',
                 color='darkblue', linewidth=2)
        ax2.set_ylim(-20, 20)
        ax2.set_xlabel('Cut lateral  [mm]')
        ax2.set_ylabel('Height [mm]')
        ax2.spines['right'].set_bounds(-25, 25)
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_label_position("right")
        plt.grid(which='major', linestyle='-',)
        plt.minorticks_on()
        ax2.set_yticks([-20, -10, 0, 10, 20])
        ax2.set_aspect('equal')
        ax3 = plt.subplot2grid((5, 7), (0, 4), colspan=3, rowspan=3)
        plt.plot([-1, 65], [0, 0], 'k', linewidth=0.5)
        plt.plot(y_s[-220,:], h_s[-220,:], ':',
                 color='darkblue', linewidth=2.5)
        # plt.plot(np.arange(0, h.shape[1]), h_s[int(x.shape[1]-12), :], ':',
        #         color='darkblue', linewidth=1.5)
        #  -np.mean(h_s[int(x.shape[1]-12), :]),
        ax3.set_ylim(-5, 5)
        # ax3.set_xticks(np.arange(0, 130, 25))
        ax3.set_xlabel('Cut longitudal [mm]')
        #  ax3.set_ylabel('Deformation to mean [mm]')
        ax3.set_ylabel('Height [mm]')
        ax3.spines['right'].set_bounds(-20, 20)
        ax3.spines['left'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.yaxis.set_ticks_position('right')
        ax3.xaxis.set_ticks_position('bottom')
        ax3.yaxis.set_label_position("right")
        plt.grid(which='major', linestyle='-',)
        plt.minorticks_on()
        ax3.set_yticks([-20, -10, 0, 10, 20])
        ax3.set_aspect('equal')
        plt.tight_layout()
        svg_file = self.path+'/plot/{:05d}.svg'.format(int(name))
        png_file = self.path+'/plot/{:05d}.png'.format(int(name))
        # plt.savefig(svg_file)
        plt.savefig(png_file)
        plt.close()

    def _get_name(self, kind):
        if hasattr(self, "file_name"):
            return self.file_name[:-3] + "_light.h5"

        serie = self.file_name

        str_ind0 = serie._compute_strindices_from_indices(
            *[inds[0] for inds in serie.get_index_slices()]
        )

        str_ind1 = serie._compute_strindices_from_indices(
            *[inds[1] - 1 for inds in serie.get_index_slices()]
        )

        name = (
            "surface_tracking" + serie.base_name + str_ind0 + "-" + str_ind1 + "_light.h5"
        )
        return name

    def save(self,name):
        pathsav = "../../../surfacetracking/111713/results/"
        name = "aName"+str(SurfaceTrackingObject.i)
        SurfaceTrackingObject.i += 1
        scipy.io.savemat(pathsav + name,
                         mdict={'H_sav': self.H_sav, 'H_filt': self.H_filt})

    #
    # def save(self, path=None, out_format="uvmat", kind=None):
    #     path = '../../../surfacetracking/111713/results/'
    #     # name = self._get_name(kind)
    #     name = "name"
    #
    #     if path is not None:
    #         path_file = os.path.join(path, name)
    #     else:
    #         path_file = name
    #
    #     with h5py.File(path_file, "w") as f:
    #         f.attrs["class_name"] = "Surface_tracking"
    #         f.attrs["module_name"] = "fluidimage.data_objects.surface_tracking"
    #
    #         self._save_in_hdf5_object(f, tag="sf")
    #
    #     return self
    #
    # def _save_in_hdf5_object(self, f, tag="sf"):
    #
    #     if "class_name" not in f.attrs.keys():
    #         f.attrs["class_name"] = "Surface_tracking"
    #         f.attrs["module_name"] = "fluidimage.data_objects.surface_tracking"
    #     if "params" not in f.keys():
    #         self.params._save_as_hdf5(hdf5_parent=f)
    #     # if "couple" not in f.keys():
    #     #     self.couple.save(hdf5_parent=f)
    #
    #     g_piv = f.create_group(tag)
    #     g_piv.attrs["class_name"] = "Surface_tracking"
    #     g_piv.attrs["module_name"] = "fluidimage.data_objects.surface_tracking"
    #
    #     for k in self._keys_to_be_saved:
    #         g_piv.create_dataset(k, data=self.__dict__[k])