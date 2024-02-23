from time import perf_counter

from path_images import get_path

from fluidimage.piv import Work

params = Work.create_default_params()

params.piv0.shape_crop_im0 = 64
params.piv0.grid.overlap = 0.5

params.multipass.number = 3
params.multipass.use_tps = False

params.fix.displacement_max = 3
params.fix.correl_min = 0.1
params.fix.threshold_diff_neighbour = 3

params.series.path = str(get_path("2005C") / "c*.bmp")
params.series.str_subset = "i, 0:2"
params.series.ind_start = 48

work = Work(params=params)

t0 = perf_counter()
piv = work.process_1_serie()
t1 = perf_counter()
print(f"Work done in {t1 - t0:.3f} s.")

piv.display(show_interp=False, scale=0.1, show_error=True)
