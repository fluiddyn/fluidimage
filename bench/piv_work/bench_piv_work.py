
import cProfile
import pstats

from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV

params = WorkPIV.create_default_params()

# for a very short computation
params.piv0.shape_crop_im0 = 32
params.piv0.grid.overlap = 0.5

# params.piv0.method_subpix = 'centroid'
# params.piv0.method_correl = 'theano'

params.multipass.number = 2
params.multipass.use_tps = 'last'
params.multipass.coeff_zoom = [2]

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)


cProfile.runctx('result = piv.calcul(serie)',
                globals(), locals(), 'profile.pstats')

s = pstats.Stats('profile.pstats')
s.strip_dirs().sort_stats('time').print_stats(10)

print(
    'with gprof2dot and graphviz (command dot):\n'
    'gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png')
