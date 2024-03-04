"""Post-processing of vector fields (:mod:`fluidimage.postproc.vector_field`)
=============================================================================

.. autoclass:: VectorFieldOnGrid
   :members:
   :private-members:

.. autoclass:: ArrayOfVectorFieldsOnGrid
   :members:
   :private-members:

"""

import itertools
from copy import deepcopy
from numbers import Number
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from fluiddyn.util.paramcontainer import ParamContainer

from .util import (
    compute_1dspectrum,
    compute_2dspectrum,
    compute_div,
    compute_rot,
    reshape_on_grid_final,
)


def _is_regular(x):
    dx = x[1] - x[0]
    return np.allclose(np.diff(x), dx)


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
        if isinstance(z, np.ndarray) and z.ndim == 0:
            z = z.item()

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

        self.is_grid_regular = _is_regular(x) and _is_regular(y)

        if z is not None and self.is_grid_regular:
            self.is_grid_regular = (
                isinstance(z, Number) or z.shape == vx.shape or _is_regular(z)
            )

        # data orientation for proper quiver
        flipx = self.x[1] - self.x[0] < 0
        flipy = self.y[1] - self.y[0] < 0
        if flipx:
            self.x = np.flip(self.x)
            self.vx = np.fliplr(self.vx)
            self.vy = np.fliplr(self.vy)
            if not (isinstance(self.vz, Number) or not vz.shape == vx.shape):
                self.vz = np.fliplr(self.vz)
            if z is not None and not (
                isinstance(self.z, Number) or not self.vx.shape == self.z.shape
            ):
                self.z = np.fliplr(self.z)
        if flipy:
            self.y = np.flip(self.y)
            self.vx = np.flipud(self.vx)
            self.vy = np.flipud(self.vy)
            if not (isinstance(self.vz, Number) or not vz.shape == vx.shape):
                self.vz = np.flipud(self.vz)
            if z is not None and not (
                isinstance(self.z, Number) or not self.vx.shape == self.z.shape
            ):
                self.z = np.flipud(self.z)

        if self.is_grid_regular:
            self.dx = self.x[1] - self.x[0]
            self.dy = self.y[1] - self.y[0]

    @property
    def shape(self):
        """Shape of the field (``(nz, ny, nx)`` or ``(ny, nx)``)"""
        if self.z is None or isinstance(self.z, Number) or self.z.ndim > 1:
            return len(self.y), len(self.x)
        else:
            return len(self.z), len(self.y), len(self.x)

    def save(self, path, params=None):
        """Save the object in a hdf5 file"""
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
            class_name in ("MultipassPIVResults", "LightPIVResults")
            and module_name == "fluidimage.data_objects.piv"
        ):
            with h5py.File(path, "r") as file:
                params = ParamContainer(hdf5_object=file["params"])

            if class_name == "MultipassPIVResults":
                key_piv = f"/piv{params.multipass.number-1}"
            else:
                key_piv = "piv"

            with h5py.File(path, "r") as file:
                piv = file[key_piv]
                ixvecs_final = piv["ixvecs_final"][...]
                iyvecs_final = piv["iyvecs_final"][...]
                deltaxs = piv["deltaxs_final"][...]
                deltays = piv["deltays_final"][...]

                kwargs = {}
                (
                    X,
                    Y,
                    kwargs["vx"],
                    kwargs["vy"],
                ) = reshape_on_grid_final(
                    ixvecs_final, iyvecs_final, deltaxs, deltays
                )

                kwargs["x"] = X[0, :]
                kwargs["y"] = Y[:, 0]

                kwargs["z"] = None
                kwargs["params"] = params

        elif class_name == cls.__name__ and module_name == cls.__module__:
            kwargs = {}
            with h5py.File(path, "r") as file:
                for attr_name in cls._attr_saved_as_dataset:
                    kwargs[attr_name] = file[attr_name][...]
                for attr_name in cls._attr_saved_as_attr:
                    attr = file.attrs[attr_name]
                    if isinstance(attr, np.ndarray):
                        attr = list(attr)
                    kwargs[attr_name] = attr

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

    def display(
        self, scale=1, background=None, ax=None, skip=(slice(None), slice(None))
    ):
        """Display the vector field"""

        if background is not None:
            raise NotImplementedError

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        pcm = ax.pcolormesh(self.x, self.y, self.compute_norm())

        # minus because the y axis is inverted
        q = ax.quiver(
            self.x[skip[0]],
            self.y[skip[1]],
            self.vx[skip],
            self.vy[skip],
            scale_units="xy",
            scale=scale,
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
        # y axis inverted!
        # ax.set_ylim([ymax + ly / n, ymin - ly / n])
        ax.set_ylim([ymin - ly / n, ymax + ly / n])
        ax.set_aspect("equal")

        return ax

    def median_filter(self, size, niter=1, valid=True):
        """Return a new field filtered with a median filter"""
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
            f"median_filter(size={size}, niter={niter}, valid ={valid})"
        )

        return ret

    def gaussian_filter(self, sigma, niter=1, truncate=3, valid=True):
        """Return a new field filtered with a gaussian filter"""
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
        """Return a new field extrated from `self`"""

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
        if (
            self.z is not None
            and not isinstance(self.z, Number)
            and self.vx.shape == self.z.shape
        ):
            ret.z = _extract2d(ret.z)

        ret.history.append(
            (
                "extract(start0={}, stop0={}, " "start1={}, stop1={}, phys={})"
            ).format(start0, stop0, start1, stop1, phys)
        )

        return ret

    def truncate(self, cut=1, phys=False):
        """Return a new truncated field"""

        if phys:
            raise NotImplementedError

        ny, nx = self.vx.shape
        return self.extract(cut, ny - cut, cut, nx - cut)

    def extract_square(self, cut=0, force_even=True):
        """Return a square field"""
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
        """Compute the norm of the vector field"""
        return np.sqrt(self.vx**2 + self.vy**2)

    def compute_spatial_fft(self, axes=(0, 1)):
        """Compute the spatial Fourier transform"""
        if axes == (0, 1):
            vx_fft, kx, ky, psd_vx = compute_2dspectrum(
                self.x, self.y, self.vx, axes=axes
            )
            vy_fft, kx, ky, psd_vy = compute_2dspectrum(
                self.x, self.y, self.vy, axes=axes
            )
            return vx_fft, vy_fft, kx, ky, psd_vx, psd_vy
        elif axes == 0:
            vx_fft, ky, psd_vx = compute_1dspectrum(self.y, self.vx, axis=axes)
            vy_fft, ky, psd_vy = compute_1dspectrum(self.y, self.vy, axis=axes)
            return vx_fft, vy_fft, self.x, ky, psd_vx, psd_vy
        elif axes == 1:
            vx_fft, kx, psd_vx = compute_1dspectrum(self.x, self.vx, axis=axes)
            vy_fft, kx, psd_vy = compute_1dspectrum(self.x, self.vy, axis=axes)
            return vx_fft, vy_fft, kx, self.y, psd_vx, psd_vy

    def compute_rotz(self, edge_order=2):
        """Compute the vertical curl"""

        if not self.is_grid_regular:
            raise NotImplementedError

        dvx_dy = np.gradient(self.vx, self.dy, axis=0, edge_order=edge_order)
        dvy_dx = np.gradient(self.vy, self.dx, axis=1, edge_order=edge_order)
        return compute_rot(dvx_dy, dvy_dx)

    def compute_divh(self, edge_order=2):
        """Compute the horizontal divergence"""

        if not self.is_grid_regular:
            raise NotImplementedError

        dvx_dx = np.gradient(self.vx, self.dx, axis=1, edge_order=edge_order)
        dvy_dy = np.gradient(self.vy, self.dy, axis=0, edge_order=edge_order)
        return compute_div(dvx_dx, dvy_dy)


class ArrayOfVectorFieldsOnGrid:
    def __init__(self, fields=None):
        if fields is None:
            fields = []
        elif not isinstance(fields, (list, tuple)):
            raise TypeError

        self._list = list(fields)

        self.timestep = None
        self.times = None
        self.unit_time = None

    def set_timestep(self, timestep, unit_time="sec"):
        """Set the timestep (and a time vector)"""
        self.timestep = timestep
        self.unit_time = unit_time
        nb_files = len(self)
        self.times = np.linspace(0, timestep * (nb_files - 1), nb_files)

    def compute_time_average(self):
        """Compute the time average"""
        result = deepcopy(self._list[0])
        for piv in self._list[1:]:
            result += piv
        return result / len(self._list)

    def append(self, v):
        """Append an element"""
        self._list.append(v)

    def extend(self, l):
        """Extend from an iterable"""
        self._list.extend(l)

    def __add__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v + other)
        return result

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v - other)
        return result

    def __mul__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v * other)
        return result

    def __rmul__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(other * v)
        return result

    def __truediv__(self, other):
        result = type(self)()
        for v in self._list:
            result.append(v / other)
        return result

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, key):
        r = self._list.__getitem__(key)
        if isinstance(r, list):
            r = type(self)(r)
        return r

    def __setitem__(self, key, value):
        self._list.__setitem__(key, value)

    def __delitem__(self, key):
        self._list.__delitem__(key)

    def __repr__(self):
        return (
            "ArrayPIV containing {} fields:\n".format(len(self))
            + self._list.__repr__()
        )

    def __len__(self):
        return self._list.__len__()

    def median_filter(self, size, niter=1, valid=True):
        """Return a new array filtered with a median filter"""
        result = type(self)()
        for v in self:
            result.append(v.median_filter(size, niter=niter, valid=valid))
        return result

    def gaussian_filter(self, sigma, niter=1, truncate=3, valid=True):
        """Return a new array filtered with a gaussian filter"""
        result = type(self)()
        for v in self:
            result.append(
                v.gaussian_filter(
                    sigma, niter=niter, truncate=truncate, valid=valid
                )
            )
        return result

    def extract(self, start0, stop0, start1, stop1, phys=False):
        """Extract new fields from the fields"""
        result = type(self)()
        for v in self:
            result.append(v.extract(start0, stop0, start1, stop1, phys=phys))
        return result

    def truncate(self, cut=1, phys=False):
        """Truncate the fields"""
        result = type(self)()
        for v in self:
            result.append(v.truncate(cut=cut, phys=phys))
        return result

    def extract_square(self, cut=0):
        """Extract square fields"""
        result = type(self)()
        for v in self:
            result.append(v.extract_square(cut=cut))
        return result

    def compute_temporal_fft(self):
        """Compute the temporal Fourier transform"""
        if self.times is None:
            raise RuntimeError(
                "please use `set_timestep` to define time before performing "
                "temporal Fourier transform"
            )

        piv0 = self._list[0]
        fields = np.empty([len(self._list), *piv0.shape])
        for it, piv in enumerate(self._list):
            fields[it, :, :] = piv.vx

        vx_fft, omega, psdU = compute_1dspectrum(self.times, fields, axis=0)

        for it, piv in enumerate(self._list):
            fields[it, :, :] = piv.vy

        vy_fft, omega, psdV = compute_1dspectrum(self.times, fields, axis=0)

        return vx_fft, vy_fft, omega, psdU, psdV

    def apply_function_to_spatiotemp_data(self, func):
        """
        Apply the function func on 3D data vx(t, y, x) and vy(t, y, x).
        """
        piv0 = self._list[0]
        fields = np.empty([len(self._list), *piv0.shape])
        for it, piv in enumerate(self._list):
            fields[it, :, :] = piv.vx
        retvx = func(fields)
        for it, piv in enumerate(self._list):
            fields[it, :, :] = piv.vy
        retvy = func(fields)
        result = deepcopy(self)
        for it, v in enumerate(result):
            v.vx = retvx[it]
            v.vy = retvy[it]
        return result

    def display(
        self,
        ind=0,
        scale=1,
        background=None,
        ax=None,
        skip=(slice(None), slice(None)),
    ):
        ax = self._list[ind].display(scale, background, ax, skip)
        fig = ax.figure
        self.currentind = ind

        def onscroll(event):
            if event.button == "up":
                self.currentind = (self.currentind + 1) % len(self)
            else:
                self.currentind = (self.currentind - 1) % len(self)
            C = self._list[self.currentind].compute_norm()
            ax.collections[0].set_array(C.ravel())
            ax.collections[1].set_UVC(
                self._list[self.currentind].vx[skip],
                self._list[self.currentind].vy[skip],
            )
            print(f"t ={self.times[self.currentind]:.2f} " + self.unit_time)
            fig.canvas.draw()

        fig.canvas.mpl_connect("scroll_event", onscroll)
        return ax
