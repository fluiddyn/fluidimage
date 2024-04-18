"""Smooth 2d fields
===================


"""

import itertools

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import convolve

from fluidimage.calcul.interpolate.griddata import griddata

weights = np.ones([3, 3])


def _smooth(a, is_correct):
    # assert a.shape == is_correct.shape
    norm = convolve(is_correct, weights, mode="nearest")
    ind = np.where(norm == 0)
    norm[ind] = 1
    return convolve(a, weights, mode="nearest") / norm


def smooth_clean(xs, ys, deltaxs, deltays, iyvecs, ixvecs, threshold):
    """Smooth and clean the displacements

    Consider the nan values and a threshold for the difference between the vector
    and its neighbors.

    Warning: important for perf (~40% for PIV)

    """
    nx = len(ixvecs)
    ny = len(iyvecs)

    shape = [ny, nx]
    arr_is_not_nan = np.ones(shape)

    selection = ~(np.isnan(deltaxs) | np.isnan(deltays))
    if not selection.any():
        return deltaxs, deltays

    centers = np.vstack([xs[selection], ys[selection]])
    dxs_select = deltaxs[selection]
    dys_select = deltays[selection]
    dxs = griddata(centers, dxs_select, (ixvecs, iyvecs)).reshape(shape)
    dys = griddata(centers, dys_select, (ixvecs, iyvecs)).reshape(shape)

    assert arr_is_not_nan.shape == dxs.shape

    dxs2 = _smooth(dxs, arr_is_not_nan)
    dys2 = _smooth(dys, arr_is_not_nan)

    indices = (abs(dxs2 - dxs) + abs(dys2 - dys) > threshold).nonzero()

    dxs[indices] = 0
    dys[indices] = 0
    arr_is_not_nan[indices] = 0

    dxs_smooth = _smooth(dxs, arr_is_not_nan)
    dys_smooth = _smooth(dys, arr_is_not_nan)

    # come back to the unstructured grid
    xy = list(itertools.product(ixvecs, iyvecs))
    interpolator_x = LinearNDInterpolator(xy, dxs_smooth.T.flat)
    interpolator_y = LinearNDInterpolator(xy, dys_smooth.T.flat)
    out_dxs = interpolator_x(xs, ys)
    out_dys = interpolator_y(xs, ys)

    return out_dxs, out_dys
