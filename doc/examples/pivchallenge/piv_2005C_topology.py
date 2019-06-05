
import os

from fluidimage.topologies.piv import TopologyPIV
from path_images import get_path

path = os.path.join(get_path('2005C'), 'c*.bmp')

params = TopologyPIV.create_default_params()

params.series.path = path
params.series.strcouple = 'i, 0:2'
params.series.ind_start = 48
params.series.ind_stop = 52

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = False

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3


params.saving.how = 'complete'

topology = TopologyPIV(params)

serie = topology.series.serie

topology.compute()
