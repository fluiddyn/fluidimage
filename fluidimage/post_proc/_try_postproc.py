from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
import numpy as np
import h5py

params = WorkPIV.create_default_params()

# for a very short computation
params.piv0.shape_crop_im0 = 24
params.piv0.grid.overlap = 0.

params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'fftw'

params.multipass.number = 2
# params.multipass.use_tps = True

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()

result.save()




# calculate tps coeff
centers = np.vstack([x, y])
smoothing_coef = 0
subdom_size = 20

tps = ThinPlateSplineSubdom(
    centers, subdom_size, smoothing_coef,
    threshold=1, pourc_buffer_area=0.5)

U_smooth, U_tps = tps.compute_tps_coeff_subdom(U)
V_smooth, V_tps = tps.compute_tps_coeff_subdom(V)

# interpolation grid
xI = yI = np.arange(0, 2*pi, 0.1)
XI, YI = np.meshgrid(xI, yI)
XI = XI.ravel()
YI = YI.ravel()

new_positions = np.vstack([XI, YI])

tps.init_with_new_positions(new_positions)

U_eval = tps.compute_eval(U_tps)
V_eval = tps.compute_eval(V_tps)
