"""Griddata wrapper function
============================


"""

import numpy as np
from matplotlib import mlab
from scipy import interpolate


def griddata(centers, values, new_positions, using="scipy"):
    xnew = new_positions[0]
    ynew = new_positions[1]
    grid_x, grid_y = np.meshgrid(xnew, ynew)

    if using == "scipy":
        # 'linear' or 'cubic'
        values_new = interpolate.griddata(
            centers.T, values, (grid_x, grid_y), "cubic"
        )
    elif using == "matplotlib":
        x = centers[0]
        y = centers[1]
        values_new = mlab.griddata(x, y, values, xnew, ynew, "linear")
        values_new[values_new.mask] = np.nan
    else:
        raise ValueError

    inds = np.where(np.isnan(values_new))
    values_nearest = interpolate.griddata(
        centers.T, values, (grid_x, grid_y), "nearest"
    )
    values_new[inds] = values_nearest[inds]

    return values_new.flatten()
