from path_images import get_path

from fluidimage.piv import Topology

params = Topology.create_default_params()

params.series.path = str(get_path("2001A") / "A*")
params.series.str_subset = "i, 1:3"
params.series.ind_start = 1

params.piv0.shape_crop_im0 = 32

params.multipass.number = 2
params.multipass.use_tps = False

params.saving.how = "recompute"

topology = Topology(params)

serie = topology.series.serie

topology.compute()
