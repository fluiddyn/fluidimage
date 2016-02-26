
import numpy as np

import matplotlib.pyplot as plt

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

from fluidimage.calcul.interpolate.thin_plate_spline import \
    compute_tps_coeff, ThinPlateSpline

params = WorkPIV.create_default_params()
params.piv0.shape_crop_im0 = 16
params.piv0.grid.overlap = 0.
params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'fftw'
piv = WorkPIV(params=params)

series = SeriesOfArrays('../../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

# calcul the piv field
result = piv.calcul(serie)

# for the interpolation
selection = ~np.isnan(result.deltaxs)

xs = result.xs[selection]
ys = result.ys[selection]

deltaxs = result.deltaxs[selection]
deltays = result.deltays[selection]

centers = np.vstack([xs, ys])

coef = 0.5
deltaxs_smooth, deltax_tps = compute_tps_coeff(centers, deltaxs, coef)
deltays_smooth, deltay_tps = compute_tps_coeff(centers, deltays, coef)


im0, im1 = result.couple.get_arrays()
ny, nx = im0.shape

dx = 8

new_xs = np.arange(0, nx, dx)
new_ys = np.arange(0, ny, dx)

new_xs, new_ys = np.meshgrid(new_xs, new_ys)
new_xs = new_xs.ravel()
new_ys = new_ys.ravel()

new_positions = np.vstack([new_xs, new_ys])

tps = ThinPlateSpline(new_positions, centers)

deltaxs_eval = tps.compute_field(deltax_tps)
deltays_eval = tps.compute_field(deltay_tps)

fig = plt.figure()
ax = plt.gca()

q = ax.quiver(result.xs, result.ys, result.deltaxs, result.deltays, color='r')
q1 = ax.quiver(xs, ys, deltaxs_smooth, deltays_smooth, color='g')
q2 = ax.quiver(new_xs, new_ys, deltaxs_eval, deltays_eval, color='gray')

plt.show()
