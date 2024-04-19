import numpy as np
from transonic import boost


def _mean_neighbors(arr, arr_is_not_nan):
    """Nan has to be zeros in `arr`!"""

    arr_a = arr
    arr_b = arr_is_not_nan.astype(np.uint8)

    n0, n1 = arr.shape
    arr_output = np.zeros_like(arr)
    arr_nb_pts = np.zeros(arr.shape, dtype=np.uint8)

    # corners
    i0, i1 = 0, 0
    if arr_is_not_nan[i0, i1]:
        arr_output[i0, i1] = (
            arr_a[i0 + 1, i1] + arr_a[i0 + 1, i1 + 1] + arr_a[i0, i1 + 1]
        )
        arr_nb_pts[i0, i1] = (
            arr_b[i0 + 1, i1] + arr_b[i0 + 1, i1 + 1] + arr_b[i0, i1 + 1]
        )

    i0, i1 = 0, n1 - 1
    if arr_is_not_nan[i0, i1]:
        arr_output[i0, i1] = (
            arr_a[i0 + 1, i1] + arr_a[i0 + 1, i1 - 1] + arr_a[i0, i1 - 1]
        )
        arr_nb_pts[i0, i1] = (
            arr_b[i0 + 1, i1] + arr_b[i0 + 1, i1 - 1] + arr_b[i0, i1 - 1]
        )

    i0, i1 = n0 - 1, 0
    if arr_is_not_nan[i0, i1]:
        arr_output[i0, i1] = (
            arr_a[i0 - 1, i1] + arr_a[i0 - 1, i1 + 1] + arr_a[i0, i1 + 1]
        )
        arr_nb_pts[i0, i1] = (
            arr_b[i0 - 1, i1] + arr_b[i0 - 1, i1 + 1] + arr_b[i0, i1 + 1]
        )

    i0, i1 = n0 - 1, n1 - 1
    if arr_is_not_nan[i0, i1]:
        arr_output[i0, i1] = (
            arr_a[i0 - 1, i1] + arr_a[i0 - 1, i1 - 1] + arr_a[i0, i1 - 1]
        )
        arr_nb_pts[i0, i1] = (
            arr_b[i0 - 1, i1] + arr_b[i0 - 1, i1 - 1] + arr_b[i0, i1 - 1]
        )

    # upper raw
    i0 = 0
    for i1 in range(1, n1 - 1):
        if not arr_is_not_nan[i0, i1]:
            continue
        tmp_a = arr_a[i0, i1 - 1] + arr_a[i0, i1 + 1]
        tmp_b = arr_b[i0, i1 - 1] + arr_b[i0, i1 + 1]
        for idx in range(-1, 2):
            tmp_a += arr_a[i0 + 1, i1 - idx]
            tmp_b += arr_b[i0 + 1, i1 - idx]
        arr_output[i0, i1] = tmp_a
        arr_nb_pts[i0, i1] = tmp_b

    # bottom raw
    i0 = n0 - 1
    for i1 in range(1, n1 - 1):
        if not arr_is_not_nan[i0, i1]:
            continue
        tmp_a = arr_a[i0, i1 - 1] + arr_a[i0, i1 + 1]
        tmp_b = arr_b[i0, i1 - 1] + arr_b[i0, i1 + 1]
        for idx in range(-1, 2):
            tmp_a += arr_a[i0 - 1, i1 - idx]
            tmp_b += arr_b[i0 - 1, i1 - idx]
        arr_output[i0, i1] = tmp_a
        arr_nb_pts[i0, i1] = tmp_b

    # left column
    i1 = 0
    for i0 in range(1, n0 - 1):
        if not arr_is_not_nan[i0, i1]:
            continue
        tmp_a = arr_a[i0 - 1, i1] + arr_a[i0 + 1, i1]
        tmp_b = arr_b[i0 - 1, i1] + arr_b[i0 + 1, i1]
        for idx in range(-1, 2):
            tmp_a += arr_a[i0 + idx, i1 + 1]
            tmp_b += arr_b[i0 + idx, i1 + 1]
        arr_output[i0, i1] = tmp_a
        arr_nb_pts[i0, i1] = tmp_b

    # right column
    i1 = n1 - 1
    for i0 in range(1, n0 - 1):
        if not arr_is_not_nan[i0, i1]:
            continue
        tmp_a = arr_a[i0 - 1, i1] + arr_a[i0 + 1, i1]
        tmp_b = arr_b[i0 - 1, i1] + arr_b[i0 + 1, i1]
        for idx in range(-1, 2):
            tmp_a += arr_a[i0 + idx, i1 - 1]
            tmp_b += arr_b[i0 + idx, i1 - 1]
        arr_output[i0, i1] = tmp_a
        arr_nb_pts[i0, i1] = tmp_b

    # core
    for i0 in range(1, n0 - 1):
        for i1 in range(1, n1 - 1):
            if not arr_is_not_nan[i0, i1]:
                continue
            tmp_a = arr_a[i0, i1 - 1] + arr_a[i0, i1 + 1]
            tmp_b = arr_b[i0, i1 - 1] + arr_b[i0, i1 + 1]
            for idx in range(-1, 2):
                tmp_a += arr_a[i0 - 1, i1 - idx] + arr_a[i0 + 1, i1 - idx]
                tmp_b += arr_b[i0 - 1, i1 - idx] + arr_b[i0 + 1, i1 - idx]
            arr_output[i0, i1] = tmp_a
            arr_nb_pts[i0, i1] = tmp_b

    # normalize
    for i0 in range(n0):
        for i1 in range(n1):
            if not arr_is_not_nan[i0, i1] or arr_nb_pts[i0, i1] == 0:
                arr_output[i0, i1] = np.nan
                continue
            arr_output[i0, i1] /= arr_nb_pts[i0, i1]

    return arr_output


@boost
def mean_neighbors_xy(
    deltaxs: "float32[]",
    deltays: "float32[]",
    iyvecs: "int64[]",
    ixvecs: "int64[]",
):
    """Compute the mean over the nearest neighbors"""

    n1 = nx = len(ixvecs)
    n0 = ny = len(iyvecs)

    shape = (ny, nx)
    arr_is_not_nan = np.ones(shape, dtype=bool)

    deltaxs2d = deltaxs.copy().reshape(shape)
    deltays2d = deltays.copy().reshape(shape)

    for i0 in range(n0):
        for i1 in range(n1):
            if np.isnan(deltaxs2d[i0, i1]) or np.isnan(deltays2d[i0, i1]):
                arr_is_not_nan[i0, i1] = False
                deltaxs2d[i0, i1] = 0.0
                deltays2d[i0, i1] = 0.0

    mean_x = _mean_neighbors(deltaxs2d, arr_is_not_nan)
    mean_y = _mean_neighbors(deltays2d, arr_is_not_nan)
    return mean_x.reshape(n0 * n1), mean_y.reshape(n0 * n1)
