
from fluidimage import config_logging
config_logging('info')

from fluidimage.topologies.piv import TopologyPIV

params = TopologyPIV.create_default_params()

path = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/pivchallenge/PIV2001A'
params.series.path = path
params.series.strcouple = 'i, 1:3'
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = 'last'

params.saving.path = path + '/RESULT_FLUIDIMAGE2'
params.saving.how = 'recompute'

topology = TopologyPIV(params)

serie = topology.series.serie

topology.compute()
