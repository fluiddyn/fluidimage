from path_images import get_path

from fluidimage.piv import Topology

params = Topology.create_default_params()

params.series.path = str(get_path("2005C") / "c*.bmp")
# params.series.ind_stop = 12

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = "last"
params.multipass.subdom_size = 200

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3

params.saving.how = "recompute"

topology = Topology(params)

topology.compute()
# topology.compute(nb_max_workers=4)
# topology.compute("multi_exec_subproc")
