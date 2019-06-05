"""To be run in IPython to find a good set of parameters"""

from fluidimage import SeriesOfArrays
from fluidimage.works.optical_flow import WorkOpticalFlow

params = WorkOpticalFlow.create_default_params()

params.mask.strcrop = "30:250, 50:350"

# params.features._print_doc()
params.features.maxCorners = 100000
params.features.qualityLevel = 0.05
params.features.blockSize = 20

# params.optical_flow._print_doc()
params.optical_flow.maxLevel = 2
params.optical_flow.winSize = (48, 48)

work = WorkOpticalFlow(params=params)

path = "../../image_samples/Oseen/Images"
# path = '../../image_samples/Karman/Images'
series = SeriesOfArrays(path, "i+1:i+3")
serie = series.get_serie_from_index(0)

piv = work.calcul(serie)

piv.display(scale=0.3, show_error=False)
