"""This example shows how to plug user-defined functions or classes that
process an image with a fluidimage topology (which process several images).

The user has to write an importable function (or class) processing one image
(see {mod}`fluidimage.image2image`).

Here, we use a class defined in fluidimage
(`fluidimage.image2image.Im2ImExample`), but it can be any
importable class!

"""

from fluidimage.image2image import Topology

params = Topology.create_default_params()

# for a function:
# params.im2im = 'fluidimage.image2image.im2im_func_example'

# for a class (with one argument for the function init):
params.im2im = "fluidimage.image2image.Im2ImExample"
params.args_init = ((1024, 2048), "clip")

params.images.path = "../../image_samples/Jet/Images/c*"

params.saving.postfix = "im2im_example"
params.saving.how = "recompute"

topo = Topology(params)

topo.compute()

assert len(topo.results) == 4
