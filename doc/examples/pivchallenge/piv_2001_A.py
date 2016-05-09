
import os

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV


params = WorkPIV.create_default_params()

# params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'fftw'

params.piv0.shape_crop_im0 = 64
params.piv0.shape_crop_im1 = 64
# params.piv0.grid.overlap = 0.5

params.multipass.number = 2
params.multipass.use_tps = True
# params.multipass.coeff_zoom = [2, 2]

params.fix.remove_error_vec = False
params.fix.delta_max = 15
params.fix.correl_min = 0.1

piv = WorkPIV(params=params)

DIR_DATA_PIV_CHALLENGE='/fsnet/project/meige/2016/16FLUIDIMAGE/samples/pivchallenge'
#path_challenge = os.environ[DIR_DATA_PIV_CHALLENGE]
path = os.path.join(DIR_DATA_PIV_CHALLENGE, 'PIV2001A', 'A*')

series = SeriesOfArrays(path, 'i+1, 1:3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()
