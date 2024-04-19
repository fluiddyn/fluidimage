"""To be run in IPython to find a good set of parameters"""

from fluidimage.piv import Work

params = Work.create_default_params()

params.series.path = "../../image_samples/Oseen/Images"

params.mask.strcrop = "30:250, 40:"

params.piv0.shape_crop_im0 = 32
params.piv0.displacement_max = 7

params.fix.correl_min = 0.2
params.fix.threshold_diff_neighbour = 8

params.multipass.number = 2
params.multipass.use_tps = "last"

work = Work(params=params)

piv = work.process_1_serie()

# piv.display(show_interp=True, scale=0.3, show_error=True)
piv.display(show_interp=False, scale=1, show_error=True)

# piv.save()
