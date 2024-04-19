"""Griddata wrapper function
============================


"""

import numpy as np
from scipy import interpolate


def griddata(centers, values, new_positions):
    xnew = new_positions[0]
    ynew = new_positions[1]
    grid_x, grid_y = np.meshgrid(xnew, ynew)

    # 'linear' or 'cubic'
    values_new = interpolate.griddata(
        centers.T, values, (grid_x, grid_y), "linear"
    )

    inds = np.where(np.isnan(values_new))
    values_nearest = interpolate.griddata(
        centers.T, values, (grid_x, grid_y), "nearest"
    )
    values_new[inds] = values_nearest[inds]

    return values_new.flatten()
