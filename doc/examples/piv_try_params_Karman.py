"""To be run in IPython to find a good set of parameters"""

from fluidimage.piv import Work

params = Work.create_default_params()

params.series.path = "../../image_samples/Karman/Images"

params.mask.strcrop = "50:350, 0:380"

params.fix.correl_min = 0.4
params.fix.threshold_diff_neighbour = 2.0

params.piv0.shape_crop_im0 = 32
params.piv0.displacement_max = 5
params.piv0.nb_peaks_to_search = 2

params.multipass.number = 2
params.multipass.use_tps = "last"

work = Work(params=params)

piv = work.process_1_serie()

# piv.display(show_interp=True, scale=0.3, show_error=False)
piv.display(show_interp=False, scale=1, show_error=True)

# piv.save()
