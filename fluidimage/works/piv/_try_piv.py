
# import h5py

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
from fluidimage.data_objects.piv import LightPIVResults


params = WorkPIV.create_default_params()

# # for a very short computation
# params.piv0.shape_crop_im0 = 128
# params.piv0.grid.overlap = 0.

# params.piv0.method_subpix = 'centroid'
# params.piv0.method_correl = 'pythran'

params.multipass.number = 1
params.multipass.use_tps = False
# params.multipass.coeff_zoom = [2, 2]

# bug params.piv0.shape_crop_im0 = 128  # !!
params.piv0.shape_crop_im0 = 64  # (80, 90)
# params.piv0.shape_crop_im1 = (38, 36)
params.fix.correl_min = 0.2
params.fix.threshold_diff_neighbour = 4
# params.piv0.grid.overlap = 10

piv = WorkPIV(params=params)

series = SeriesOfArrays("../../../image_samples/Oseen/Images", "i+1:i+3")
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()

result.save()

# lightresult = result.make_light_result()
# lightresult.save()

# lightresultload = LightPIVResults(str_path='piv_Oseen_center01-02_light.h5')

# f=h5netcdf.File('piv_Oseen_center01-02.h5')
# f=h5py.File('piv_Oseen_center01-02.h5')
