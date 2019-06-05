

import cProfile
import os
import pstats

from fluidimage.topologies.piv import TopologyPIV
from path_images import get_path

path = os.path.join(get_path('2005C'), 'c*.bmp')

params = TopologyPIV.create_default_params()

params.series.path = path
params.series.strcouple = 'i, 0:2'
# params.series.ind_start = 48
params.series.ind_stop = 20

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = False

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3


params.saving.how = 'recompute'

topology = TopologyPIV(params)

serie = topology.series.serie

cProfile.runctx('topology.compute()',
                globals(), locals(), 'profile.pstats')

s = pstats.Stats('profile.pstats')
s.strip_dirs().sort_stats('time').print_stats(10)

print(
    'with gprof2dot and graphviz (command dot):\n'
    'gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png')
