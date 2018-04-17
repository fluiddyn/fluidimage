"""Simple 2D calibration
========================

.. autoclass:: Calibration2DSimple
   :members:
   :private-members:

"""

import numpy as np
from fluiddyn.io.image import imread


class Calibration2DSimple(object):
    """Simple 2D calibration

    The hypothesis are (at least):

    - No rotation (x and y remains x and y).
    - No optical aberration.
    - Everything orthogonal (the camera points directly toward the laser plan).

    Parameters
    ----------

    point0: sequence of size 2
      The coordinates in pixel of a first point as (indy, indx).

    point1: sequence of size 2
      The coordinates in pixel of a first point as (indy, indx).

    distance: number
      The distance between the 2 points in physical unit.

    aspect_ratio_pixel: number
      The aspect ratio of 1 pixel (a characteristic of the camera) as
      yphys1pxel/xphys1pxel.

    shape_image: sequence of size 2 (optional)
      The shape of the image. If it is not given, the next argument
      (`path_image`) is used.

    path_image: str (optional)
      The path toward the image (used to get the shape of the image).

    point_reference: sequence of size 2 (optional, default=(0, 0))
      A point for which (yphys, xphys) = orig_phys

    orig_phys: sequence of size 2 (optional, default=(0., 0.))
      The coordinates in physical unit of the point origin.

    """

    def __init__(
        self,
        point0,
        point1,
        distance,
        aspect_ratio_pixel=1.,
        shape_image=None,
        path_image=None,
        point_origin=(0, 0),
        orig_phys=(0., 0.),
    ):

        self.point0 = np.array(point0)
        self.point1 = np.array(point1)
        self.distance = distance

        delta_point = self.point1 - self.point0

        self.aspect_ratio_pixel = aspect_ratio_pixel

        self.xphys1pixel = distance / np.sqrt(
            delta_point[0] ** 2 + (aspect_ratio_pixel * delta_point[1]) ** 2
        )
        self.yphys1pixel = aspect_ratio_pixel * self.xphys1pixel

        if shape_image is None:
            arr = imread(path_image)
            shape_image = arr.shape
        self.shape_image = shape_image

        self.ix_origin = point_origin[1]
        self.iy_origin = point_origin[0]

        self.xphys_origin = orig_phys[1]
        self.yphys_origin = orig_phys[0]

    def pix2phys(self, ixs, iys):
        """Calcul physical coordinates from indices.

        Parameters
        ----------

        ixs: number or np.array
          Indices in the x direction (second dimension).

        iys: number or np.array
          Indices in the y direction (first dimension).

        """

        xphys = self.xphys_origin + self.xphys1pixel * (ixs - self.ix_origin)
        yphys = self.yphys_origin + self.yphys1pixel * (iys - self.iy_origin)
        return xphys, yphys

    def displ2phys(self, ixs, iys, displxs, displys):
        """Calcul physical coordinates and physical displacements from indices.

        Parameters
        ----------
        ixs: number or np.array
          Indices in the x direction (second dimension).

        iys: number or np.array
          Indices in the y direction (first dimension).

        displxs: number or np.array
          Displacements in the x direction (in pixels)

        displys: number or np.array
          Displacements in the y direction (in pixels)

        """
        xphys, yphys = self.pix2phys(ixs, iys)

        displ_phys_xs = self.xphys1pixel * displxs
        displ_phys_ys = self.yphys1pixel * displys

        return xphys, yphys, displ_phys_xs, displ_phys_ys
