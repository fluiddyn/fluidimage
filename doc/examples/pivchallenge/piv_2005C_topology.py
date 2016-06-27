
import os

from path_images import get_path

from fluidimage import config_logging
config_logging('info')

from fluidimage.topologies.piv import TopologyPIV

params = TopologyPIV.create_default_params()

path = os.path.join(get_path('2005C'), 'c*')

params.series.path = path
params.series.strcouple = 'i, 0:2'
params.series.ind_stop = 10

params.piv0.shape_crop_im0 = 32
params.piv0.grid.overlap = 0.5

params.multipass.number = 2
params.multipass.use_tps = True

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3


params.saving.how = 'recompute'

topology = TopologyPIV(params)

serie = topology.series.serie

topology.compute()
