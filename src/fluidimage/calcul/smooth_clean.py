"""Smooth 2d fields
===================


"""

import itertools

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import convolve

from fluidimage.calcul.interpolate.griddata import griddata

weights = np.ones([3, 3])


def _smooth(a, for_norm):
    norm = convolve(for_norm, weights, mode="nearest")
    ind = np.where(norm == 0)
    norm[ind] = 1
    return convolve(a, weights, mode="nearest") / norm


def smooth_clean(xs, ys, deltaxs, deltays, iyvecs, ixvecs, threshold):
    nx = len(ixvecs)
    ny = len(iyvecs)

    shape = [ny, nx]
    for_norm = np.ones(shape)

    selection = ~(np.isnan(deltaxs) | np.isnan(deltays))
    if not selection.any():
        return deltaxs, deltays

    centers = np.vstack([xs[selection], ys[selection]])
    dxs_select = deltaxs[selection]
    dys_select = deltays[selection]
    dxs = griddata(centers, dxs_select, (ixvecs, iyvecs)).reshape([ny, nx])
    dys = griddata(centers, dys_select, (ixvecs, iyvecs)).reshape([ny, nx])

    dxs2 = _smooth(dxs, for_norm)
    dys2 = _smooth(dys, for_norm)

    inds = (abs(dxs2 - dxs) + abs(dys2 - dys) > threshold).nonzero()

    dxs[inds] = 0
    dys[inds] = 0
    for_norm[inds] = 0

    dxs_smooth = _smooth(dxs, for_norm)
    dys_smooth = _smooth(dys, for_norm)

    # come back to the unstructured grid
    xy = list(itertools.product(ixvecs, iyvecs))
    interpolator_x = LinearNDInterpolator(xy, dxs_smooth.T.flat)
    interpolator_y = LinearNDInterpolator(xy, dys_smooth.T.flat)
    out_dxs = interpolator_x(xs, ys)
    out_dys = interpolator_y(xs, ys)

    # previous implementation (interp2d is depreciated)
    # fxs = interp2d(ixvecs, iyvecs, dxs_smooth, kind="linear")
    # fys = interp2d(ixvecs, iyvecs, dys_smooth, kind="linear")
    # out_dxs = np.empty_like(deltaxs)
    # out_dys = np.empty_like(deltays)
    # for i, (x, y) in enumerate(zip(xs, ys)):
    #     out_dxs[i] = fxs(x, y)[0]
    #     out_dys[i] = fys(x, y)[0]

    return out_dxs, out_dys
