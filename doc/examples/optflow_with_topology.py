
from fluidimage import path_image_samples
from fluidimage.topologies.optical_flow import Topology

params = Topology.create_default_params()

params.series.path = path_image_samples / 'Karman/Images'
params.series.ind_start = 1
params.series.ind_step = 2

# params.features._print_doc()
params.features.maxCorners = 100000
params.features.qualityLevel = 0.05
params.features.blockSize = 20

# params.optical_flow._print_doc()
params.optical_flow.maxLevel = 2
params.optical_flow.winSize = (48, 48)

params.mask.strcrop = ':, 50:500'

# params.saving.how = 'complete'
params.saving.postfix = 'optflow_example'

topology = Topology(params, logging_level='info')

# To produce a graph of the topology
# topology.make_code_graphviz('topo.dot')

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)
