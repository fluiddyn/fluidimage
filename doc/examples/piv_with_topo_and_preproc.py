
from fluidimage import path_image_samples
from fluidimage.topologies.piv import TopologyPIV

params = TopologyPIV.create_default_params()

params.series.path = path_image_samples / 'Karman/Images'
params.series.ind_start = 1
params.series.ind_step = 2

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = False

params.mask.strcrop = ':, 50:500'

params.saving.how = 'recompute'
params.saving.postfix = 'piv_example'

# we use the light versatile preprocessing feature:
params.preproc.im2im = 'my_example_im2im.im2im'

# Here the "image to image" function will be imported with the statement
# `from my_example_im2im import im2im`

topology = TopologyPIV(params, logging_level='info')

# To produce a graph of the topology
topology.make_code_graphviz('topo.dot')

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)
