from fluidimage import get_path_image_samples
from fluidimage.bos import Topology

params = Topology.create_default_params()

params.images.path = get_path_image_samples() / "Karman/Images"
params.images.str_subset = "1:3"

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = False

params.mask.strcrop = ":, 50:500"

params.saving.how = "recompute"
params.saving.postfix = "bos_example"

topology = Topology(params, logging_level="info")

# To produce a graph of the topology
# topology.make_code_graphviz('topo.dot')

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)

assert len(topology.results) == 1
