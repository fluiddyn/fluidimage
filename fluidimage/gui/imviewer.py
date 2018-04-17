"""Simple viewer adapted for fluid images (:mod:`fluidimage.gui.imviewer`)
==========================================================================

Coded with matplotlib GUI!

.. autoclass:: ImageViewer
   :members:
   :private-members:

"""
from __future__ import print_function, division

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt

try:
    from matplotlib.widgets import TextBox, Button

    str_error_import_textbox = False
except ImportError as error_import_textbox:
    str_error_import_textbox = (
        str(error_import_textbox)
        + "\nfluidimviewer needs matplotlib.widgets.TextBox "
        "which is included in matplotlib 2.1."
    )

    import matplotlib

    if matplotlib.__version__ < "2.1":
        print("Warning: " + str_error_import_textbox)
    else:
        raise

from fluiddyn.io.image import imread
from fluiddyn.util import time_as_str
from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles

extensions = ["png", "tif", "tiff", "jpg", "jpeg", "bmp", "cine"]
extensions = ["." + ext for ext in extensions]

debug = False


def _print_debug(*args):
    if debug:
        print(*args)


def check_image(path):
    return any([path.endswith(ext) for ext in extensions])


size_button = 0.06
sep_button = 0.02
x0 = 0.3

name_buttons = ["-n", "-1", "+1", "+n"]

x_buttons = [
    x0 + i * (size_button + sep_button) for i in range(len(name_buttons))
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path",
        help="Path file or directory.",
        type=str,
        nargs="?",
        default=os.getcwd(),
    )
    parser.add_argument(
        "-cm", "--colormap", help="colormap", type=str, default=""
    )
    parser.add_argument("-v", "--verbose", help="verbose mode", action="count")

    return parser.parse_args()


class ImageViewer(object):
    """Simple Image viewer."""

    def __init__(self, args):

        if str_error_import_textbox:
            raise ImportError(str_error_import_textbox)

        path_in = args.path
        if os.path.isdir(path_in):
            self.path_files = glob(os.path.join(path_in, "*"))
            self.path_files = [
                path for path in self.path_files if check_image(path)
            ]
            self.path_files.sort()
            ifile = 0
        else:
            path_file = glob(path_in)[0]
            self.path_files = glob(os.path.join(os.path.split(path_file)[0], "*"))
            self.path_files = [
                path for path in self.path_files if check_image(path)
            ]
            self.path_files.sort()
            ifile = self.path_files.index(path_file)

        if len(self.path_files) == 0:
            raise ValueError("No image files detected.")

        if len(self.path_files) == 1 and self.path_files[0].endswith(".cine"):
            serie = SerieOfArraysFromFiles(self.path_files[0])
            self.path_files = serie.get_path_arrays()

        path_dir = os.path.split(self.path_files[0])[0]
        self.nb_images = len(self.path_files)
        print("Will use {} images in the dir {}".format(self.nb_images, path_dir))

        self._buttons = {}
        self._textboxes = {}

        fig = self.fig = plt.figure()
        fig.canvas.set_window_title(
            path_dir + " (" + time_as_str()[-8:].replace("-", ":") + ")"
        )
        self.ax = fig.add_axes([0.07, 0.15, 0.7, 0.78])

        self.maps = {}
        try:
            self.cmap = plt.cm.viridis
        except AttributeError:
            self.cmap = plt.cm.jet

        self.ifile = ifile
        self._last_was_increase = False

        im = imread(self.path_files[ifile])
        self.clim = [0, 0.99 * im.max()]
        self._updating_clim = False

        self.loadim(ifile, im)
        name_file = self.get_namefile()
        self.ax.set_title(name_file)

        self._image_changing = False

        function_buttons = [self._do_nothing] * len(name_buttons)
        function_buttons[0] = self._decrease_ifile_n
        function_buttons[1] = self._decrease_ifile
        function_buttons[2] = self._increase_ifile
        function_buttons[3] = self._increase_ifile_n

        y = size_button / 3.
        for i, x in enumerate(x_buttons):
            name = name_buttons[i]
            func = function_buttons[i]
            self._create_button(fig, [x, y, size_button, size_button], name, func)

        self._n = 1

        self._create_text(
            fig,
            [0.1, y, 2 * size_button, size_button],
            "n = ",
            self._submit_n,
            "1",
        )

        self._create_text(
            fig,
            [0.87, 0.92, 1.5 * size_button, size_button],
            "cmax = ",
            self._change_cmax,
            "{:.2f}".format(self.clim[1]),
        )

        self._create_text(
            fig,
            [0.87, 0.1, 1.5 * size_button, size_button],
            "cmin = ",
            self._change_cmin,
            "{:.2f}".format(self.clim[0]),
        )

        self._create_button(
            fig, [0.65, 0.945, 1.2 * size_button, 0.045], "reload", self.reloadim
        )

        self._create_button(
            fig, [0.85, y, size_button, size_button], "auto", self.set_autoclim
        )

        cax = fig.add_axes([0.83, 0.2, 0.07, 0.7])
        self.cbar = fig.colorbar(self.mappable, cax=cax)

        fig.canvas.mpl_connect("key_press_event", self.onclick)
        print("press alt+h for help")

        plt.show()

    def set_autoclim(self, event):
        _print_debug("set_autoclim")
        im = imread(self.path_files[self.ifile])
        self.clim = [im.min(), 0.99 * im.max()]
        self._update_clim()

    def reloadim(self, event):
        self.loadim(self.ifile)

    def loadim(self, ifile, im=None):
        _print_debug("loadim", ifile, im)
        if im is None:
            im = imread(self.path_files[ifile])
            _print_debug(self.path_files[ifile])
            _print_debug(im)
        self.mappable = self.ax.imshow(
            im,
            interpolation="nearest",
            cmap=self.cmap,
            origin="upper",
            extent=[0, im.shape[1], im.shape[0], 0],
            vmin=self.clim[0],
            vmax=self.clim[1],
        )
        self.maps[ifile] = self.mappable

    def get_namefile(self):
        return os.path.split(self.path_files[self.ifile])[-1]

    def change_im(self):
        self.ifile = self.ifile % self.nb_images
        if self._image_changing:
            return

        self._image_changing = True
        ifile = self.ifile
        name_file = self.get_namefile()
        print("changing to file " + name_file, end="...")
        _print_debug("")
        map_old = self.mappable

        if ifile in self.maps:
            self.mappable = self.maps[ifile]
            self.mappable.set_clim(self.clim)
            self.mappable.set_visible(True)
        else:
            self.loadim(ifile)

        map_old.set_visible(False)
        self.ax.set_title(name_file)
        self.fig.canvas.draw()
        _print_debug("")
        print("\rchanged to file " + name_file + " " * 20)
        self._image_changing = False

    def _create_button(self, fig, rect, text, func):
        ax = fig.add_axes(rect)
        button = Button(ax, text)
        button.on_clicked(func)
        self._buttons[text] = button
        return button

    def _create_text(self, fig, rect, name, func, initial):
        ax = fig.add_axes(rect)
        textbox = TextBox(ax, name, initial=initial)
        textbox.on_submit(func)
        self._textboxes[name] = textbox
        return textbox

    def _switch(self):
        if self._last_was_increase:
            self._decrease_ifile_n()
        else:
            self._increase_ifile_n()
        self._last_was_increase = not self._last_was_increase

    def _do_nothing(self, event):
        print("do nothing")

    def _increase_ifile(self, event=None):
        self.ifile += 1
        self.change_im()

    def _decrease_ifile(self, event=None):
        self.ifile -= 1
        self.change_im()

    def _increase_ifile_n(self, event=None):
        self.ifile += self._n
        self.change_im()

    def _decrease_ifile_n(self, event=None):
        self.ifile -= self._n
        self.change_im()

    def _submit_n(self, text):
        self._n = eval(text, {}, {})
        print("n = ", self._n)

    def _change_cmax(self, text):
        _print_debug("_change_cmax")
        cmax = eval(text, {}, {})
        if cmax == self.clim[1]:
            return

        self.clim[1] = cmax
        self._update_clim()

    def _change_cmin(self, text):
        _print_debug("_change_cmin")
        cmin = eval(text, {}, {})

        if cmin == self.clim[0]:
            return

        self.clim[0] = cmin
        self._update_clim()

    def _update_clim(self):
        if self._updating_clim:
            return

        self._updating_clim = True
        _print_debug("_update_clim:", self.clim)
        self.mappable.set_clim(self.clim)
        self.fig.canvas.draw()
        self.cbar.set_clim(self.clim)
        self.cbar.draw_all()

        txt_cmax = self._textboxes["cmax = "]
        txt_cmin = self._textboxes["cmin = "]
        txt_cmin.set_val("{:.2f}".format(self.clim[0]))
        txt_cmax.set_val("{:.2f}".format(self.clim[1]))
        _print_debug("_update_clim end")
        self._updating_clim = False

    def onclick(self, event):
        if event.key == "alt+h":
            print("\nalt+s\t\t\t switch between images\n")

        if event.inaxes != self.ax:
            return

        if event.key == "alt+s":
            self._switch()


# if debug and __name__ == '__main__':
#     args = parse_args()
#     self = ImageViewer(args)
