from fluidimage import get_path_image_samples
from fluidimage.piv import Topology

params = Topology.create_default_params()

params.series.path = get_path_image_samples() / "Karman/Images"

params.mask.strcrop = "50:350, 0:380"

params.piv0.shape_crop_im0 = 32
params.piv0.displacement_max = 5
params.piv0.nb_peaks_to_search = 2

params.fix.correl_min = 0.4
params.fix.threshold_diff_neighbour = 2.0

params.multipass.number = 2
params.multipass.use_tps = "last"

# params.saving.how = 'complete'
params.saving.postfix = "piv_example"

topology = Topology(params, logging_level="info")

# To produce a graph of the topology
# topology.make_code_graphviz('topo.dot')

# Compute in parallel
topology.compute()
# topology.compute("multi_exec_subproc")

# Compute in sequential (for debugging)
# topology.compute(sequential=True)
