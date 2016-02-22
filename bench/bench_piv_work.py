
import pstats
import cProfile

from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

params = WorkPIV.create_default_params()

piv = WorkPIV(params=params)

series = SeriesOfArrays('../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)


cProfile.runctx('result = piv.calcul(serie)',
                globals(), locals(), 'Profile.prof')

s = pstats.Stats('Profile.prof')
s.strip_dirs().sort_stats('time').print_stats(10)
