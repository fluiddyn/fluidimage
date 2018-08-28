"""PIV post-processing
======================

.. autoclass:: PIV2d
   :members:
   :private-members:

.. autoclass:: ArrayPIV
   :members:
   :private-members:

.. autofunction:: get_grid_pixel_from_piv_file

.. autofunction:: get_grid_pixel

"""

from numbers import Number
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.ndimage.filters import median_filter, gaussian_filter

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.works.piv.multipass import WorkPIV


def get_grid_pixel_from_piv_file(path, index_pass=-1):
    """Recompute 1d arrays containing the approximate positions of the vectors

    Useful to compute a grid on which we can interpolate the displacement fields.

    Parameters
    ----------

    path: str

      Path of a PIV file.

    index_pass: int

      Index of the pass

    Returns
    -------

    xs1d: np.ndarray

      Indices (2nd, direction "x") of the pixel in the image

    ys1d: np.ndarray

      Indices (1st, direction "y") of the pixel in the image

    """
    with h5py.File(path) as f:
        params = ParamContainer(hdf5_object=f["params"])
        shape_images = f["couple/shape_images"].value

    return get_grid_pixel(params, shape_images, index_pass)


def get_grid_pixel(params, shape_images, index_pass=-1):
    """Recompute 1d arrays containing the approximate positions of the vectors

    Useful to compute a grid on which we can interpolate the displacement fields.

    Parameters
    ----------

    params: :class:`fluiddyn.util.paramcontainer.ParamContainer`

      Parameters for the class :class:`fluidimage.works.piv.multipass.WorkPIV`

    shape_images: sequence

      Shape of the images

    index_pass: int

      Index of the pass

    Returns
    -------

    xs1d: np.ndarray

      Indices (2nd, direction "x") of the pixel in the image

    ys1d: np.ndarray

      Indices (1st, direction "y") of the pixel in the image


    """

    params_default = WorkPIV.create_default_params()
    params_default._modif_from_other_params(params)

    work_multi = WorkPIV(params_default)
    work_multi._prepare_with_image(imshape=shape_images)

    work = work_multi.works_piv[index_pass]
    xs1d = work.ixvecs
    ys1d = work.iyvecs

    return xs1d, ys1d


class PIV2d(object):
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

        self.name = "Fluidimage_field"
        self.history = ["fluidimage"]
        self.setname = "??"
        self.ysign = "ysign"

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
                q.scale *= 2.
                print(key + ": multiply q.scale by 2.", end="")
            elif key == "ctrl+-":
                q.scale /= 2.
                print(key + ": divide q.scale by 2.", end="")
            if event.key in ["ctrl++", "ctrl+-"]:
                print(" q.scale = {}".format(q.scale))
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


class ArrayPIV(object):
    """Array of PIV fields on a regular grid."""

    def __init__(self, l=None):
        if l is None:
            l = []
        elif not isinstance(l, (list, tuple)):
            raise TypeError

        self._list = list(l)

    def append(self, v):
        self._list.append(v)

    def extend(self, l):
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

    def __rdiv__(self, other):
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
        self._list.__detitem__(key)

    def __repr__(self):
        return (
            "ArrayPIV containing {} fields:\n".format(len(self))
            + self._list.__repr__()
        )

    def __len__(self):
        return self._list.__len__()

    def median_filter(self, size, niter=1, valid=True):
        result = type(self)()
        for v in self:
            result.append(v.median_filter(size, niter=niter, valid=valid))
        return result

    def gaussian_filter(self, sigma, niter=1, truncate=3, valid=True):
        result = type(self)()
        for v in self:
            result.append(
                v.gaussian_filter(
                    sigma, niter=niter, truncate=truncate, valid=valid
                )
            )
        return result

    def extract(self, start0, stop0, start1, stop1, phys=False):
        result = type(self)()
        for v in self:
            result.append(v.extract(start0, stop0, start1, stop1, phys=phys))
        return result

    def truncate(self, cut=1, phys=False):
        result = type(self)()
        for v in self:
            result.append(v.truncate(cut=cut, phys=phys))
        return result

    def extract_square(self, cut=0):
        result = type(self)()
        for v in self:
            result.append(v.extract_square(cut=cut))
        return result
