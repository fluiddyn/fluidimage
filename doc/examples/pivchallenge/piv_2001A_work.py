from path_images import get_path

from fluidimage.piv import WorkPIV

params = WorkPIV.create_default_params()

params.piv0.shape_crop_im0 = 128
params.piv0.grid.overlap = 0.5

params.multipass.number = 2
params.multipass.use_tps = False

params.fix.displacement_max = 15
params.fix.correl_min = 0.1

params.series.path = str(get_path("2001A") / "A*")
params.series.str_subset = "i, 1:3"
params.series.ind_start = 1

piv = WorkPIV(params=params)

result = piv.process_1_serie()

result.display()
