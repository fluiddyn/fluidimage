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


import h5py


from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.works.piv.multipass import WorkPIV
from .vector_field import VectorFieldOnGrid


class PIV2d(VectorFieldOnGrid):
    pass


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
    with h5py.File(path, "r") as file:
        params = ParamContainer(hdf5_object=file["params"])
        shape_images = file["couple/shape_images"][...]

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


class ArrayPIV:
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
