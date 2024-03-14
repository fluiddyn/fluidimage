from path_images import get_path

from fluidimage.piv import Topology

params = Topology.create_default_params()

params.series.path = str(get_path("2005C") / "c*.bmp")
# params.series.ind_stop = 12

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = False

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3

params.saving.how = "complete"

topology = Topology(params)

topology.compute()
# topology.compute("multi_exec_subproc")
