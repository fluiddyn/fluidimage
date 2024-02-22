import os

from fluidimage.piv import Topology

params = Topology.create_default_params()

params.series.path = "../../../image_samples/Karman/Images3"
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

params.saving.how = "recompute"
params.saving.postfix = "old_piv"

topology = Topology(params, logging_level="info")
# topology.make_code_graphviz('topo.dot')

topology.compute()
