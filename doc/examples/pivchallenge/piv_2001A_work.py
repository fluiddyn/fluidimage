
import os

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
from path_images import get_path

params = WorkPIV.create_default_params()

params.piv0.shape_crop_im0 = 128
params.piv0.grid.overlap = 0.5

params.multipass.number = 2
params.multipass.use_tps = False

params.fix.displacement_max = 15
params.fix.correl_min = 0.1

piv = WorkPIV(params=params)

path = os.path.join(get_path('2001A'), 'A*')

series = SeriesOfArrays(path, 'i, 1:3', ind_start=1)
serie = series.get_serie_from_index(1)

result = piv.calcul(serie)

result.display()
