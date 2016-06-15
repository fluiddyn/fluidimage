
from fluidimage import SeriesOfArrays
from fluidimage.works.piv import WorkPIV


params = WorkPIV.create_default_params()

params.multipass.number = 2
params.multipass.use_tps = False

params.piv0.shape_crop_im0 = 32
params.fix.correl_min = 0.2
params.fix.threshold_diff_neighbour = 8

work = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

piv = work.calcul(serie)

piv.display(show_interp=False, scale=0.05, show_error=True)

# result.save()
