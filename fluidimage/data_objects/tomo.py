"""Tomography data objects (:mod:`fluidimage.data_objects.tomo`)
================================================================

.. autoclass:: ArrayTomoBase
   :members:
   :private-members:

.. autoclass:: ArrayTomo
   :members:
   :private-members:

.. autoclass:: ArrayTomoCV
   :members:
   :private-members:

"""


import os
import shutil

import dask.array as da
import h5py
import matplotlib.pyplot as plt
import numpy as np

try:
    import ipyvolume.pylab as p3
except ImportError:
    print(
        "Install ipyvolume to visualize ArrayTomo in 3D in a Jupyter widget." ""
    )


class ArrayTomoBase:
    """Data structure to hold the tomographic data for a single instant.

    """

    _keys_to_save = ["xs", "ys", "zs", "I"]
    _attrs_to_save = ["nx", "ny", "nz"]

    def __init__(
        self,
        xlims=(0, 10),
        ylims=(0, 10),
        zlims=(0, 5),
        nb_voxels=(10, 10, 5),
        image_path=None,
        params=None,
        h5file_path=None,
        dtype=np.float64,
    ):
        """Initialize

        Parameters
        ----------
        xlims, ylims, zlims : tuple of int
            Limits of the 3D physical volume geometry

        nb_voxels : tuple of int or array-like
            Shape of the voxel array in the three axes, ie. (nx, ny, nz)

        image_path : str
            Reference image filename

        dtype : numpy floating point datatype
            Control the precision of the arrays

        """

        self.xmin, self.xmax = xlims
        self.ymin, self.ymax = ylims
        self.ymin, self.zmax = zlims
        self.nb_voxels = nb_voxels
        self.dtype = dtype
        if h5file_path is not None:
            self.load(h5file_path)
        else:
            self.nx, self.ny, self.nz = nb_voxels

            def grid1d(lims, nb):
                start, stop = lims
                return np.linspace(start, stop, nb, endpoint=True, dtype=dtype)

            self.xs, self.ys, self.zs = map(
                grid1d, (xlims, ylims, zlims), nb_voxels
            )

        if image_path is not None:
            self.init_paths(image_path)
        self.params = params

    def init_paths(self, image_path, output_dir=None):
        self.image_path = image_path
        if output_dir is None:
            output_dir = os.path.abspath(
                os.path.join(os.path.dirname(image_path), "..", "tomo")
            )
        self.h5file_path = os.path.join(output_dir, self._get_name())

    def describe(self, vmin=None, vmax=None):
        """Describe the voxel intensity array."""
        I = self.I
        if vmin is not None:
            x, y = np.where(I < vmin)
            print(f"No. of points below {vmin}={x.size}")
        if vmax is not None:
            x, y = np.where(I > vmax)
            print(f"No. of points above {vmax}={x.size}")
        string = (
            f" min={I.min()}\n max={I.max()}\n mean={I.mean()}\n median={np.median(I)}\n "
            f" std={I.std()}\n shape={I.shape}"
        )
        string = [s.split("=") for s in string.splitlines()]
        final_string = []
        for l, r in string:
            final_string.append(" = ".join((l.rjust(10), r)))
        print("\n".join(final_string))

    def clear(self):
        """Reset intensities of all voxels as unity."""
        if isinstance(self.I, np.ndarray):
            self.I[:] = 1
        else:
            print(f"Warning: Not clearing I of type {type(self.I)}")
        if os.path.exists(self.h5file_path):
            print(f"rm {self.h5file_path}")
            os.remove(self.h5file_path)

    def load(self, h5file_path=None):
        if h5file_path is None:
            h5file_path = self.h5file_path
        else:
            self.h5file_path = h5file_path

        with h5py.File(h5file_path) as f:
            grp = f["tomo"]
            for k in self._keys_to_save:
                setattr(self, k, grp[k][...])
            for attr in self._attrs_to_save:
                setattr(self, attr, grp.attrs[attr])

    def load_dataset(self, h5file_path=None, tag="tomo", key="I", copy=False):
        if h5file_path is None:
            if copy:
                h5file_path, ext = os.path.splitext(self.h5file_path)
                h5file_path += "_tmp" + ext
                print(f"Copying {self.h5file_path} to {h5file_path}...", end="")
                shutil.copyfile(self.h5file_path, h5file_path)
                print("Done.")
            else:
                h5file_path = self.h5file_path

        f = h5py.File(h5file_path)
        try:
            grp = f[tag]
            dset = grp[key]
        except KeyError:
            print(
                h5file_path,
                "contains\n Groups:",
                list(f),
                "and",
                tag,
                "contains\n Datasets:",
                list(grp),
            )
            f.close()
            raise
        return f, dset, h5file_path

    def _get_name(self):
        return os.path.splitext(os.path.basename(self.image_path))[0] + ".h5"

    def save(self, path=None, sparse=False):
        if path is None:
            path = self.h5file_path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        if sparse:
            raise NotImplementedError("Save the intensity values as sparse")

        with h5py.File(path, "w") as f:
            self._save_in_hdf5_object(f)

    def _save_in_hdf5_object(self, f, tag="tomo"):
        if "class_name" not in f.attrs.keys():
            f.attrs["class_name"] = self.__class__.__name__
            f.attrs["module_name"] = self.__module__
        if "params" not in f.keys() and self.params is not None:
            self.params._save_as_hdf5(hdf5_parent=f)

        if tag in f:
            grp = f[tag]
        else:
            grp = f.create_group(tag)

        grp.attrs["class_name"] = self.__class__.__name__
        grp.attrs["module_name"] = self.__module__
        for attr in self._attrs_to_save:
            grp.attrs[attr] = getattr(self, attr)

        for k in self._keys_to_save:
            data = getattr(self, k)
            print(f"Saving {type(data)} {k}...")
            if isinstance(data, da.core.Array):
                dataset = grp.require_dataset(
                    f"/{tag}/{k}", shape=data.shape, dtype=data.dtype
                )
                da.store(data, dataset)
            else:
                grp.create_dataset(k, data=data)

    def plot3d(self, threshold=0.5):
        """Display the reconstructed intensities as an Jupyter widget using
        ipyvolume.

        """
        if isinstance(self.I, da.core.Array):
            raise ValueError("Cannot display dask arrays.")
        sel = self.I > threshold
        print(f"Displaying {np.where(sel)[0].size} points...")
        fig = p3.figure()
        cmap = plt.cm.gray
        color = cmap(self.I[sel] / self.I.max())
        try:
            s = p3.scatter(
                self.X[sel],
                self.Y[sel],
                self.Z[sel],
                color=color,
                marker="sphere",
            )
        except AttributeError:
            grid_sel = self.grid[sel]
            s = p3.scatter(
                grid_sel[:, 0],
                grid_sel[:, 1],
                grid_sel[:, 2],
                color=color,
                marker="sphere",
            )
        s.size = self.I[sel] / self.I.max() * 5
        p3.view(90, 0)

        return fig, s

    def plot_slices(self, start=0, stop=None):
        I = self.I
        if I.ndim == 1:
            nx, ny, nz = map(len, (self.xs, self.ys, self.zs))
            I = I.reshape((nx, ny, nz))

        if stop is None:
            stop = I.shape[2]

        plt.figure()
        im = plt.imshow(I[:, :, start])
        plt.colorbar(im)
        for i2 in range(start + 1, stop):
            im.set_data(I[:, :, i2])
            plt.draw()
            plt.pause(0.5)


class ArrayTomo(ArrayTomoBase):
    """A typical 3D meshgrid for tomographic reconstruction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X, self.Y, self.Z = np.meshgrid(self.xs, self.ys, self.zs)

        # Intensities
        if "h5file_path" not in kwargs:
            self.I = np.ones_like(self.X)


class ArrayTomoCV(ArrayTomoBase):
    """A flattened 3D array for tomographic reconstruction. This format is
    suitable for OpenCV based algorithms.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grid = (
            np.vstack(np.meshgrid(self.zs, self.ys, self.xs, indexing="ij"))
            .reshape(3, -1)
            .T
        )

        # Swap columns to make it back to x, y ,z
        self.grid = np.empty_like(grid)
        self.grid[:, 0] = grid[:, 2]
        self.grid[:, 1] = grid[:, 1]
        self.grid[:, 2] = grid[:, 0]

        if "h5file_path" not in kwargs:
            self.I = np.ones(len(self.grid))
