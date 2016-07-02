from fluidimage.topologies.piv import TopologyPIV

from fluidimage import config_logging
config_logging('info')


params = TopologyPIV.create_default_params()

params.series.path = '../../image_samples/Karman/Images'
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'complete'
params.saving.postfix = 'piv_complete'

topology = TopologyPIV(params)

topology.compute()
