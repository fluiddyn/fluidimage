import numpy as np

from thin_plate_spline import (
    compute_tps_matrix,
    compute_tps_matrix_numpy,
    compute_tps_matrix_pythran,
)

n0 = 259

n1 = 1000

x = np.random.rand(n0)
y = np.random.rand(n0)

centers = np.vstack([x, y])

x = np.random.rand(n1)
y = np.random.rand(n1)

new_positions = np.vstack([x, y])


"""

%timeit compute_tps_matrix(new_positions, centers)
%timeit compute_tps_matrix_numpy(new_positions, centers)

%timeit compute_tps_matrix_pythran(new_positions, centers)

"""
