
import numpy as np

from scipy.ndimage import convolve
from scipy.interpolate import interp2d

from fluidimage.calcul.interpolate.griddata import griddata

weights = np.ones([3, 3])


def _smooth(a, for_norm):
    norm = convolve(for_norm, weights, mode='nearest')
    return convolve(a, weights, mode='nearest') / norm


def smooth_clean(xs, ys, deltaxs, deltays, iyvecs, ixvecs, threshold):

    nx = len(ixvecs)
    ny = len(iyvecs)

    shape = [ny, nx]
    for_norm = np.ones(shape)

    selection = ~np.isnan(deltaxs)
    centers = np.vstack([ys[selection], xs[selection]])
    dxs = deltaxs[selection]
    dys = deltays[selection]

    dxs = griddata(centers, dxs, (iyvecs, ixvecs)).reshape(shape)
    dys = griddata(centers, dys, (iyvecs, ixvecs)).reshape(shape)

    dxs2 = _smooth(dxs, for_norm)
    dys2 = _smooth(dys, for_norm)

    inds = (abs(dxs2 - dxs) + abs(dys2 - dys) > threshold).nonzero()

    dxs[inds] = 0
    dys[inds] = 0
    for_norm[inds] = 0

    dxs = _smooth(dxs, for_norm)
    dys = _smooth(dys, for_norm)

    # come back to the unstructured grid
    fxs = interp2d(ixvecs, iyvecs, dxs, kind='linear')
    fys = interp2d(ixvecs, iyvecs, dys, kind='linear')

    out_dxs = np.empty_like(deltaxs)
    out_dys = np.empty_like(deltays)

    for i, (x, y) in enumerate(zip(xs, ys)):
        out_dxs[i] = fxs(x, y)[0]
        out_dys[i] = fys(x, y)[0]

    return out_dxs, out_dys