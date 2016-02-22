

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

params = WorkPIV.create_default_params()

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)
