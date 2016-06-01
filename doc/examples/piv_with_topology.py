
from fluidimage.topologies.piv import TopologyPIV

from fluidimage import config_logging
config_logging('info')


params = TopologyPIV.create_default_params()

params.series.path = '../../image_samples/Karman/Images'
params.series.ind_start = 1
params.series.ind_step = 2

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

# params.saving.how = 'complete'
params.saving.postfix = 'piv_example'

topology = TopologyPIV(params)

# To produce a graph of the topology
# topology.make_code_graphviz('topo.dot')

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)
