

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

params = WorkPIV.create_default_params()

# for a very short computation
params.piv0.shape_crop_im0 = 16
params.piv0.grid.overlap = 0.

params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'fftw'

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()
