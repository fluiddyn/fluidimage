"""Griddata wrapper function
============================


"""
import numpy as np

from matplotlib import mlab
from scipy import interpolate


def griddata(centers, values, new_positions, using="scipy"):
    xnew = new_positions[1]
    ynew = new_positions[0]
    grid_y, grid_x = np.meshgrid(ynew, xnew)

    if using == "scipy":
        # 'linear' or 'cubic'
        values_new = interpolate.griddata(
            centers.T, values, (grid_y, grid_x), "cubic"
        )
    elif using == "matplotlib":
        x = centers[1]
        y = centers[0]
        values_new = mlab.griddata(y, x, values, ynew, xnew, "linear")
        values_new[values_new.mask] = np.nan
    else:
        raise ValueError

    inds = np.where(np.isnan(values_new))
    values_nearest = interpolate.griddata(
        centers.T, values, (grid_y, grid_x), "nearest"
    )
    values_new[inds] = values_nearest[inds]

    return values_new.flatten()
