
import os

from fluidimage.topologies.piv import TopologyPIV
from path_images import get_path

params = TopologyPIV.create_default_params()

path = os.path.join(get_path('2001A'), 'A*')

params.series.path = path
params.series.strcouple = 'i, 1:3'
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32

params.multipass.number = 2
params.multipass.use_tps = False

params.saving.how = 'recompute'

topology = TopologyPIV(params)

serie = topology.series.serie

topology.compute()
