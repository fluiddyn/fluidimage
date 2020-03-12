"""Post-processing of vector fields
===================================

.. autoclass:: VectorFieldOnGrid
   :members:
   :private-members:

"""

from copy import deepcopy
from numbers import Number
import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter, median_filter

from fluiddyn.util.paramcontainer import ParamContainer

from .util import reshape_on_grid_final


class VectorFieldOnGrid:
    """PIV field on a regular grid.

    Parameters
    ----------

    x : np.array (1d)

    y : np.array (1d)

    z : number or np.array (1d)

    vx : np.array (2d)

    vy : np.array (2d)

    vz : np.array (2d), optional

    namevx : str, 'vx'
    namevy : str, 'vy'
    namevz : str, 'vz'

    unitvx : str, '?'
    unitvy : str, '?'
    unitvz : str, '?'

    namex : str, 'x'
    namey : str, 'y'
    namez : str, 'z'

    unitx : str, '?'
    unity : str, '?'
    unitz : str, '?'

    """

    _attr_saved_as_dataset = list(
        "".join(t) for t in itertools.product(["v", ""], "xyz")
    )
    _attr_saved_as_attr = list(
        "".join(t)
        for t in itertools.product(["name", "unit"], _attr_saved_as_dataset)
    ) + ["name", "history"]

    def __init__(
        self,
        x,
        y,
        z,
        vx,
        vy,
        vz=np.nan,
        namevx="vx",
        namevy="vy",
        namevz="vz",
        unitvx="?",
        unitvy="?",
        unitvz="?",
        namex="x",
        namey="y",
        namez="z",
        unitx="?",
        unity="?",
        unitz="?",
        name="Fluidimage_field",
        history=["fluidimage"],
        params: Optional[ParamContainer] = None,
    ):

        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.unitvx = unitvx
        self.namevx = namevx
        self.unitvy = unitvy
        self.namevy = namevy
        self.unitvz = unitvz
        self.namevz = namevz
        self.unitx = unitx
        self.namex = namex
        self.unity = unity
        self.namey = namey
        self.unitz = unitz
        self.namez = namez

        self.name = name
        self.history = history
        self.params = params

    def save(self, path, params=None):
        with h5py.File(path, "w") as file:
            cls = self.__class__
            file.attrs["class_name"] = cls.__name__
            file.attrs["module_name"] = cls.__module__

            for attr_name in self._attr_saved_as_dataset:
                file.create_dataset(attr_name, data=getattr(self, attr_name))

            for attr_name in self._attr_saved_as_attr:
                file.attrs[attr_name] = getattr(self, attr_name)

            if params is None and self.params is not None:
                params = self.params

            if params is not None:
                params._save_as_hdf5(hdf5_parent=file)

    @classmethod
    def from_file(cls, path, load_params=False):
        """Create a PIV2d object from a file

        It can be a file representing a LightPIVResults or a PIV2d object.

        """

        with h5py.File(path, "r") as file:
            class_name = file.attrs["class_name"]
            module_name = file.attrs["module_name"]

        if isinstance(class_name, bytes):
            class_name = class_name.decode()
            module_name = module_name.decode()

        if (
            class_name == "MultipassPIVResults"
            and module_name == "fluidimage.data_objects.piv"
        ):
            with h5py.File(path, "r") as file:
                params = ParamContainer(hdf5_object=file["params"])
                piv = file[f"/piv{params.multipass.number-1}"]
                ixvecs_final = piv["ixvecs_final"][...]
                iyvecs_final = piv["iyvecs_final"][...]
                deltaxs = piv["deltaxs_final"][...]
                deltays = piv["deltays_final"][...]

                kwargs = {}
                (
                    kwargs["x"],
                    kwargs["y"],
                    kwargs["vx"],
                    kwargs["vy"],
                ) = reshape_on_grid_final(
                    ixvecs_final, iyvecs_final, deltaxs, deltays
                )

                kwargs["z"] = None
                kwargs["params"] = params

        elif class_name == cls.__name__ and module_name == cls.__module__:
            kwargs = {}
            with h5py.File(path, "r") as file:
                for attr_name in cls._attr_saved_as_dataset:
                    kwargs[attr_name] = file[attr_name][...]
                for attr_name in cls._attr_saved_as_attr:
                    kwargs[attr_name] = file.attrs[attr_name]

                if "params" in file.keys():
                    params = ParamContainer(hdf5_object=file["params"])
                    kwargs["params"] = params

        else:
            print(class_name, module_name)
            raise NotImplementedError
        return cls(**kwargs)

    def __add__(self, other):
        if isinstance(other, Number):
            vx = self.vx + other
            vy = self.vy + other
            vz = self.vz + other
        else:
            vx = self.vx + other.vx
            vy = self.vy + other.vy
            vz = self.vz + other.vz

        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __radd__(self, other):
        if other == 0:
            return self

        else:
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            vx = self.vx - other
            vy = self.vy - other
            vz = self.vz - other
        else:
            vx = self.vx - other.vx
            vy = self.vy - other.vy
            vz = self.vz - other.vz

        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __mul__(self, other):
        if isinstance(other, Number):
            vx = other * self.vx
            vy = other * self.vy
            vz = other * self.vz
        else:
            vx = other.vx * self.vx
            vy = other.vy * self.vy
            vz = other.vz * self.vz
        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, Number):
            vx = self.vx / other
            vy = self.vy / other
            vz = self.vz / other
        else:
            vx = self.vx / other.vx
            vy = self.vy / other.vy
            vz = self.vz / other.vz
        return type(self)(self.x, self.y, self.z, vx, vy, vz)

    def __truediv__(self, other):
        return self.__div__(other)

    def display(self, scale=1, background=None, ax=None):

        if background is not None:
            raise NotImplementedError

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        pcm = ax.pcolormesh(self.x, self.y, self.compute_norm())

        q = ax.quiver(
            self.x, self.y, self.vx, self.vy, scale_units="xy", scale=scale
        )
        ax.set_xlabel(self.namex + " [" + self.unitx + "]")
        ax.set_ylabel(self.namey + " [" + self.unity + "]")

        def onclick(event):
            key = event.key
            if key == "ctrl++":
                q.scale *= 2.0
                print(key + ": multiply q.scale by 2.", end="")
            elif key == "ctrl+-":
                q.scale /= 2.0
                print(key + ": divide q.scale by 2.", end="")
            if event.key in ["ctrl++", "ctrl+-"]:
                print(f" q.scale = {q.scale}")
                fig.canvas.draw()

        fig.canvas.mpl_connect("key_press_event", onclick)
        plt.colorbar(pcm, ax=ax)

        xmin = self.x.min()
        xmax = self.x.max()
        lx = xmax - xmin
        ymin = self.y.min()
        ymax = self.y.max()
        ly = ymax - ymin
        n = 20
        ax.set_xlim([xmin - lx / n, xmax + lx / n])
        ax.set_ylim([ymin - ly / n, ymax + ly / n])
        ax.set_aspect("equal")

        return ax

    def median_filter(self, size, niter=1, valid=True):
        if np.isscalar(size):
            size = [size, size]

        def _medianf(f):
            for i in range(niter):
                f = median_filter(f, size)
            return f

        ret = deepcopy(self)
        ret.vx = _medianf(self.vx)
        ret.vy = _medianf(self.vy)

        if hasattr(self, "vz"):
            if isinstance(self.vz, Number):
                ret.vz = self.vz
            else:
                ret.vz = _medianf(self.vz)

        if valid:
            mf = int(np.floor(max(size) / 2))
            ny, nx = self.vx.shape
            ret = ret.extract(mf, ny - mf, mf, nx - mf)

        ret.history.append(
            "median_filter(size={}, niter={}, valid ={})".format(
                size, niter, valid
            )
        )

        return ret

    def gaussian_filter(self, sigma, niter=1, truncate=3, valid=True):
        if np.isscalar(sigma):
            sigma = [sigma, sigma]

        def _gaussianf(f):
            for i in range(niter):
                f = gaussian_filter(f, sigma, truncate=truncate)
            return f

        ret = deepcopy(self)
        ret.vx = _gaussianf(self.vx)
        ret.vy = _gaussianf(self.vy)

        if hasattr(self, "vz"):
            # ret.vz = _gaussianf(self.vz)
            pass

        if valid:
            mf = int(np.floor((2 * int(truncate * max(sigma) + 0.5) + 1) / 2))
            ny, nx = self.vx.shape
            ret = ret.extract(mf, ny - mf, mf, nx - mf)

        ret.history.append(
            "gaussian_filter(sigma={}, niter={}, valid={})".format(
                sigma, niter, valid
            )
        )

        return ret

    def extract(self, start0, stop0, start1, stop1, phys=False):
        ret = deepcopy(self)

        if phys:
            indy = (ret.y >= start0) & (ret.y <= stop0)
            indx = (ret.x >= start1) & (ret.x <= stop1)
            start0 = np.argwhere(ret.y == ret.y[indy].min())[0][0]
            stop0 = np.argwhere(ret.y == ret.y[indy].max())[0][0] + 1
            start1 = np.argwhere(ret.x == ret.x[indx].min())[0][0]
            stop1 = np.argwhere(ret.x == ret.x[indx].max())[0][0] + 1

        def _extract2d(f):
            return f[start0:stop0, start1:stop1]

        ret.x = ret.x[start1:stop1]
        ret.y = ret.y[start0:stop0]
        ret.vx = _extract2d(ret.vx)
        ret.vy = _extract2d(ret.vy)
        if hasattr(self, "vz") and np.size(self.vz) > 1:
            ret.vz = _extract2d(ret.vz)

        ret.history.append(
            (
                "extract(start0={}, stop0={}, " "start1={}, stop1={}, phys={})"
            ).format(start0, stop0, start1, stop1, phys)
        )

        return ret

    def truncate(self, cut=1, phys=False):
        if phys:
            raise NotImplementedError

        ny, nx = self.vx.shape
        return self.extract(cut, ny - cut, cut, nx - cut)

    def extract_square(self, cut=0, force_even=True):
        n1 = self.x.size
        n0 = self.y.size
        n = min(n0, n1) - 2 * cut

        if force_even and n % 2 == 1:
            n -= 1

        if n1 > n0:
            start0 = cut
            stop0 = cut + n
            start1 = (n1 - n) // 2
            stop1 = start1 + n
        else:
            start1 = cut
            stop1 = cut + n
            start0 = (n0 - n) // 2
            stop0 = start1 + n

        return self.extract(start0, stop0, start1, stop1)

    def compute_norm(self):
        return np.sqrt(self.vx ** 2 + self.vy ** 2)
