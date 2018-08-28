"""Utilities for calibration (:mod:`fluidimage.calibration.util`)
=================================================================

"""

import re
import numpy as np
from math import sin, cos
from fluiddyn.util.paramcontainer import ParamContainer, tidy_container


def get_number_from_string(string):
    return [float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", string)]


def get_number_from_string2(string):
    return [float(s) for s in string.split()]


def get_plane_equation(z0, alpha, beta):
    """Return coefficients a, b, c, d of the equation of a plane

    The plane is defined with ax + by + cz + d = 0

    Parameters
    ----------

    z0 : number

    alpha : number

      The angle in radian around x axis

    beta : number

      The angle in radian around y axis

    Notes
    -----

    Works only when 0 or 1 angle ~= 0
    """
    if alpha != 0 and beta != 0:
        raise ValueError("Works only when 0 or 1 angle != 0")

    a = sin(beta)
    b = -sin(alpha) * cos(beta)
    c = cos(alpha) * cos(beta)
    d = -c * z0
    return a, b, c, d


def get_base_from_normal_vector(nx, ny, nz):
    """matrix of base change from a given plane to the fixed plane
    n has to be approximately vertical: i.e. nz approx. 1
    """

    ez = np.array([nx, ny, nz])
    ez = ez / np.linalg.norm(ez)

    ex1, ex2 = 1, 0
    ex3 = -(ex1 * nx + ex2 * ny) / nz
    ex = np.array([ex1, ex2, ex3])
    ex = ex / np.linalg.norm(ex)

    ey = np.cross(ez, ex)
    A = np.vstack([ex, ey, ez]).transpose()
    return A, np.linalg.inv(A)


def make_params_calibration(path_file):
    """Make a tidy parameter container from a UVmat file.

    """
    params = ParamContainer(tag="calib")

    calib_uvmat = ParamContainer(path_file=str(path_file))
    tidy_container(calib_uvmat)

    calib_uvmat = calib_uvmat["geometry_calib"]

    f = [float(n) for n in np.array(get_number_from_string(calib_uvmat.fx_fy))]
    C = np.array(get_number_from_string(calib_uvmat.cx__cy))
    kc = np.array(calib_uvmat.kc)
    T = np.array(get_number_from_string(calib_uvmat.tx__ty__tz))

    R = []
    for i in range(3):
        R = np.hstack(
            [R, get_number_from_string(calib_uvmat["r_{}".format(i + 1)])]
        )

    omc = np.array(get_number_from_string(calib_uvmat["omc"]))

    params._set_attribs({"f": f, "C": C, "kc": kc, "T": T, "R": R, "omc": omc})

    if calib_uvmat.nb_slice is not None:

        nb_slice = np.array(calib_uvmat["nb_slice"])
        zslice_coord = np.zeros([nb_slice, 3])

        if calib_uvmat.nb_slice == 1:
            zslice_coord[:] = get_number_from_string(calib_uvmat["slice_coord"])
            if (
                hasattr(calib_uvmat, "slice_angle")
                and calib_uvmat["slice_angle"] is not None
            ):
                slice_angle = np.zeros([nb_slice, 3])
                slice_angle[:] = get_number_from_string(
                    calib_uvmat["slice_angle"]
                )
            else:
                slice_angle = [0, 0, 0]
        else:
            for i in range(nb_slice):
                zslice_coord[i][:] = get_number_from_string(
                    calib_uvmat["slice_coord_{}".format(i + 1)]
                )

            if (
                hasattr(calib_uvmat, "slice_angle_1")
                and calib_uvmat["slice_angle_1"] is not None
            ):
                slice_angle = np.zeros([nb_slice, 3])
                for i in range(nb_slice):
                    slice_angle[i][:] = get_number_from_string(
                        calib_uvmat["slice_angle_{}".format(i + 1)]
                    )
            else:
                slice_angle = [0, 0, 0]

        params._set_child(
            "slices",
            attribs={
                "nb_slice": nb_slice,
                "zslice_coord": zslice_coord,
                "slice_angle": slice_angle,
            },
        )

    if hasattr(calib_uvmat, "refraction_index"):
        params._set_attrib("refraction_index", calib_uvmat.refraction_index)

    if hasattr(calib_uvmat, "interface_coord"):
        params._set_attrib(
            "interface_coord",
            get_number_from_string(calib_uvmat["interface_coord"]),
        )

    return params
