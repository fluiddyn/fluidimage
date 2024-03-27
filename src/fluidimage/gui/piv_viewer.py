"""PIV viewer

Coded with matplotlib GUI!

.. autoclass:: VectorFieldsViewer
   :members:
   :private-members:

"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.gui.base_matplotlib import AppMatplotlibWidgets

size_button = 0.06
sep_button = 0.02
x0 = 0.5

name_buttons = ["-1", "+1"]

x_buttons = [
    x0 + i * (size_button + sep_button) for i in range(len(name_buttons))
]


def get_piv_data_from_path(path):

    with h5py.File(path, "r") as file:
        class_name = file.attrs["class_name"]
        module_name = file.attrs["module_name"]

    if isinstance(class_name, bytes):
        class_name = class_name.decode()
        module_name = module_name.decode()

    if (
        class_name in ("MultipassPIVResults", "LightPIVResults")
        and module_name == "fluidimage.data_objects.piv"
    ):
        if class_name == "MultipassPIVResults":
            with h5py.File(path, "r") as file:
                keys = sorted(key for key in file.keys() if key.startswith("piv"))
            key_piv = keys[-1]
        else:
            key_piv = "piv"

        with h5py.File(path, "r") as file:
            piv = file[key_piv]
            ixvecs = piv["ixvecs_final"][...]
            iyvecs = piv["iyvecs_final"][...]
            deltaxs = piv["deltaxs_final"][...]
            deltays = piv["deltays_final"][...]

    else:
        raise NotImplementedError

    return deltaxs, deltays, ixvecs, iyvecs


class VectorFieldsViewer(AppMatplotlibWidgets):
    """A simple vector field viewer."""

    @classmethod
    def parse_args(cls):

        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "path",
            help="Path file or directory.",
            type=str,
            nargs="?",
            default=os.getcwd(),
        )
        parser.add_argument(
            "-v", "--verbose", help="verbose mode", action="count"
        )

        return parser.parse_args()

    def __init__(self, args):
        super().__init__()

        path = Path(args.path).absolute()

        if path.is_file():
            path_file_init = path
        else:
            path_file_init = None

        if path.is_dir():
            try:
                path = next(path.glob("piv*"))
            except StopIteration:
                print(f"No PIV files found in {args.path}")
                sys.exit(1)

        serie = self.serie = SerieOfArraysFromFiles(path)

        if not serie:
            print(f"Not PIV file found (from {args.path=})")
            sys.exit(1)

        self.path_dir = Path(serie.path_dir)

        # initialize the figure
        fig = self.fig = plt.figure()
        self.ax = fig.add_axes([0.07, 0.15, 0.9, 0.78])

        self._init_name_files()

        if path_file_init is None:
            path_file_init = self.path_dir / self.name_files[0]
            self.index_file = 0
        else:
            self.index_file = self.name_files.index(path_file_init.name)

        self._init_figure(path_file_init)

        plt.show()

    def _init_name_files(self):

        self.name_files = sorted(
            p.name
            for p in self.path_dir.glob(
                self.serie.base_name + "*" + self.serie.extension_file
            )
        )
        self.num_files = len(self.name_files)
        self.fig.canvas.manager.set_window_title(
            f"{self.num_files} PIV files in {self.path_dir}"
        )

    def _init_figure(self, path_file_init):

        self.ax.set_title(path_file_init.name)

        deltaxs, deltays, ixvecs, iyvecs = get_piv_data_from_path(path_file_init)

        self.ax.invert_yaxis()

        self._quiver = self.ax.quiver(
            ixvecs, iyvecs, deltaxs, deltays, angles="xy", scale_units="xy"
        )

        y = size_button / 3.0

        self._create_text_box(
            self.fig,
            [0.1, y, size_button, size_button],
            "index = ",
            self._change_index_from_textbox,
            str(self.index_file),
        )

        function_buttons = [self._decrement_index, self._increment_index]
        for x, name, func in zip(x_buttons, name_buttons, function_buttons):
            self._create_button(
                self.fig, [x, y, size_button, size_button], name, func
            )

    def _increment_index(self, event=None):
        self._set_index(self.index_file + 1)

    def _decrement_index(self, event=None):
        self._set_index(self.index_file - 1)

    def _change_index_from_textbox(self, text):
        try:
            index = int(text)
        except ValueError:
            self._set_textbox_value(self.index_file)
            return
        if index >= self.num_files:
            index = self.num_files - 1
        elif index < 0:
            index = 0
        self._set_textbox_value(index)
        self._set_index(index)

    def _set_textbox_value(self, value):
        textbox = self.get_textbox("index = ")
        textbox.set_val(str(value))

    def _set_index(self, index):
        if index == self.index_file:
            return
        self.index_file = index % self.num_files
        self._set_textbox_value(self.index_file)
        self._update_fig()

    def _update_fig(self):
        path_file = self.path_dir / self.name_files[self.index_file]
        deltaxs, deltays, ixvecs, iyvecs = get_piv_data_from_path(path_file)
        self.ax.set_title(path_file.name)
        self._quiver.set_UVC(deltaxs, deltays)


def main():
    args = VectorFieldsViewer.parse_args()
    return VectorFieldsViewer(args)
