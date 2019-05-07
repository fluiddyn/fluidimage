from fluidimage import SeriesOfArrays
from fluidimage.works.optical_flow import WorkOpticalFlow


params = WorkOpticalFlow.create_default_params()

# params.mask.strcrop = '30:250, 100:'

work = WorkOpticalFlow(params=params)

path = "../image_samples/Oseen/Images"
# path = '../image_samples/Karman/Images'
series = SeriesOfArrays(path, "i+1:i+3")
serie = series.get_serie_from_index(0)

from fluidimage.data_objects.piv import ArrayCouple

couple = ArrayCouple(serie=serie)

results = work.calcul(couple)

xs, ys, displacements = results

from opyf.Render import opyfQuiverPointCloudColored
import numpy as np

positions = np.vstack((xs, ys)).T


from fluidimage.data_objects.piv import HeavyPIVResults

piv = HeavyPIVResults(
    deltaxs=displacements[:, 0].copy(),
    deltays=displacements[:, 1],
    xs=xs,
    ys=ys,
    couple=couple,
)
piv.display()

# It is needed to get a nice figure with the vortices.
# There is a bug somewhere, but where?
displacements[:, 0] = -displacements[:, 0]

opyfQuiverPointCloudColored(positions, displacements)

# piv.display(show_interp=True, scale=0.3, show_error=True)
# piv.display(show_interp=False, scale=1, show_error=True)

# result.save()
