from math import log

import numpy as np

# pythran export compute_subpix_2d_gaussian2(float32[:, :], int, int)
# pythran export compute_subpix_2d_gaussian3(float32[:, :], int, int)

# pythran export simpler_leak(float32[:, :], int, int)
# pythran export simpler_no_leak(float32[:, :], int, int)


def compute_subpix_2d_gaussian2(correl, ix, iy):

    # returning this view leads to a leak!
    correl_crop = correl[iy - 1 : iy + 2, ix - 1 : ix + 2]

    tmp = np.where(correl_crop < 0)
    for i0, i1 in zip(tmp[0], tmp[1]):
        correl_crop[i0, i1] = 1e-6

    c10 = 0
    c01 = 0
    c11 = 0
    c20 = 0
    c02 = 0
    for i in range(3):
        for j in range(3):
            c10 += (i - 1) * np.log(correl_crop[j, i])
            c01 += (j - 1) * np.log(correl_crop[j, i])
            c11 += (i - 1) * (j - 1) * np.log(correl_crop[j, i])
            c20 += (3 * (i - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c02 += (3 * (j - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c00 = (5 - 3 * (i - 1) ** 2 - 3 * (j - 1) ** 2) * np.log(
                correl_crop[j, i]
            )

    c00, c10, c01, c11, c20, c02 = (
        c00 / 9,
        c10 / 6,
        c01 / 6,
        c11 / 4,
        c20 / 6,
        c02 / 6,
    )
    deplx = (c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11**2)
    deply = (c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11**2)
    return deplx, deply, correl_crop


def compute_subpix_2d_gaussian3(correl, ix, iy):
    correl_crop = np.ascontiguousarray(correl[iy - 1 : iy + 2, ix - 1 : ix + 2])

    for i0 in range(-1, 2):
        for i1 in range(-1, 2):
            if correl_crop[i0, i1] < 0:
                correl_crop[i0, i1] = 1e-6

    c10 = 0
    c01 = 0
    c11 = 0
    c20 = 0
    c02 = 0
    for i0 in range(3):
        for i1 in range(3):
            c10 += (i1 - 1) * log(correl_crop[i0, i1])
            c01 += (i0 - 1) * log(correl_crop[i0, i1])
            c11 += (i1 - 1) * (i0 - 1) * log(correl_crop[i0, i1])
            c20 += (3 * (i1 - 1) ** 2 - 2) * log(correl_crop[i0, i1])
            c02 += (3 * (i0 - 1) ** 2 - 2) * log(correl_crop[i0, i1])
            c00 = (5 - 3 * (i1 - 1) ** 2 - 3 * (i0 - 1) ** 2) * log(
                correl_crop[i0, i1]
            )

    c00, c10, c01, c11, c20, c02 = (
        c00 / 9,
        c10 / 6,
        c01 / 6,
        c11 / 4,
        c20 / 6,
        c02 / 6,
    )
    deplx = (c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11**2)
    deply = (c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11**2)
    return deplx, deply, correl_crop


def simpler_leak(correl, ix, iy):
    # returning this view leads to a leak!
    correl_crop = correl[iy - 1 : iy + 2, ix - 1 : ix + 2]
    return correl_crop


def simpler_no_leak(correl, ix, iy):
    correl_crop = np.ascontiguousarray(correl[iy - 1 : iy + 2, ix - 1 : ix + 2])
    return correl_crop
