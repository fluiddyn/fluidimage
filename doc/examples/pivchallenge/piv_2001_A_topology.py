from fluidimage.topologies.piv import TopologyPIV
from fluidimage import config_logging
config_logging('info')

params = TopologyPIV.create_default_params()

params.series.path = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/pivchallenge/PIV2001A'

params.series.ind_start = 1
params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

params.saving.path='/fsnet/project/meige/2016/16FLUIDIMAGE/samples/pivchallenge/PIV2001A/RESULT_FLUIDIMAGE'
params.saving.postfix = 'PIV_2001A'

topology = TopologyPIV(params)

topology.compute(sequential=False)
