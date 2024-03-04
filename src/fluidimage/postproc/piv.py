"""PIV post-processing (:mod:`fluidimage.postproc.piv`)
=======================================================

.. autofunction:: get_grid_pixel_from_piv_file

.. autofunction:: get_grid_pixel

.. autoclass:: PIV2d
   :members:
   :private-members:

.. autoclass:: ArrayPIV
   :members:
   :private-members:

"""

import h5py

from fluiddyn.util.paramcontainer import ParamContainer
from fluidimage.works.piv.multipass import WorkPIV

from .vector_field import ArrayOfVectorFieldsOnGrid, VectorFieldOnGrid


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

    params: fluiddyn.util.paramcontainer.ParamContainer

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


class ArrayPIV(ArrayOfVectorFieldsOnGrid):
    """Array of PIV fields on a regular grid."""
