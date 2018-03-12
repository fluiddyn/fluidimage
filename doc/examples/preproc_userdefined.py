"""This example shows how to plug user-defined functions or classes that
process an image with a fluidimage topology (which process several images).

The user has to write an importable function (or class) processing one image
(see :mod:`fluidimage.topologies.image2image`).

"""

from fluidimage.topologies.image2image import TopologyImage2Image as Topo

params = Topo.create_default_params()

# for a function:
# params.im2im = 'fluidimage.topologies.image2image.im2im_func_example'

# for a class (with one argument for the function init):
params.im2im = 'fluidimage.topologies.image2image.Im2ImExample'
params.args_init = ((1024, 2048), 'clip')

params.series.path = '../../image_samples/Jet/Images/c*'
params.series.ind_start = 60

params.saving.how = 'recompute'

topo = Topo(params)

topo.compute()
