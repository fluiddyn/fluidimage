
import os
from time import time

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
from path_images import get_path

path = os.path.join(get_path('2005C'), 'c*.bmp')

params = WorkPIV.create_default_params()

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = False

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3

work = WorkPIV(params=params)

series = SeriesOfArrays(path, 'i, 0:2')
serie = series.get_serie_from_index(50)

t0 = time()
piv = work.calcul(serie)
t1 = time()
print('Work done in {:.3f} s.'.format(t1 - t0))

piv.display(show_interp=False, scale=0.1, show_error=True)
