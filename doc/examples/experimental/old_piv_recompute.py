import os

from fluidimage.topologies.piv import TopologyPIV

params = TopologyPIV.create_default_params()

params.series.path = '../../../image_samples/Karman/Images3'
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

params.saving.how = 'recompute'
params.saving.postfix = 'old_piv'

topology = TopologyPIV(params, logging_level='info')
#topology.make_code_graphviz('topo.dot')

topology.compute()
