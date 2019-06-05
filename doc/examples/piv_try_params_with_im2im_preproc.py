
from fluidimage import SeriesOfArrays
from fluidimage.preproc.image2image import apply_im2im_filter
from fluidimage.works.piv import WorkPIV

params = WorkPIV.create_default_params()

params.multipass.number = 2
params.multipass.use_tps = True

params.piv0.shape_crop_im0 = 32
params.piv0.displacement_max = 5
params.fix.correl_min = 0.2
params.fix.threshold_diff_neighbour = 8

params.mask.strcrop = '30:250, 100:'

work = WorkPIV(params=params)

path = '../../image_samples/Oseen/Images'
# path = '../../image_samples/Karman/Images'
series = SeriesOfArrays(path, 'i+1:i+3')
serie = series.get_serie_from_index(0)

# "image to image" filter
serie = apply_im2im_filter(serie, im2im='my_example_im2im.im2im')

piv = work.calcul(serie)

# piv.display(show_interp=True, scale=0.3, show_error=True)
piv.display(show_interp=False, scale=1, show_error=True)

# result.save()
