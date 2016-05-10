
# import h5py

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
from fluidimage.data_objects.piv import LightPIVResults


params = WorkPIV.create_default_params()

# # for a very short computation
# params.piv0.shape_crop_im0 = 128
# params.piv0.grid.overlap = 0.

# params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'pythran'

params.multipass.number = 2
params.multipass.use_tps = True
# params.multipass.coeff_zoom = [2, 2]

params.piv0.shape_crop_im0 = (80, 90)
params.piv0.shape_crop_im1 = (38, 36)
#params.fix.correl_min = 0.3
# params.piv0.grid.overlap = 10

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()

result.save()

lightresult = result.make_light_result()

# LightPIVResults(
#     result.piv1.deltaxs_approx, result.piv1.deltays_approx,
#     result.piv1.ixvecs_grid, result.piv1.iyvecs_grid,
#     couple=result.piv1.couple,
#     params=result.piv1.params)

lightresult.save()

lightresultload = LightPIVResults(str_path='piv_Oseen_center01-02_light.h5')


#f=h5netcdf.File('piv_Oseen_center01-02.h5')
#f=h5py.File('piv_Oseen_center01-02.h5')
