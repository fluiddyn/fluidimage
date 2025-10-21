import skimage

from fluidimage.topologies.surface_tracking import Topology


def rescale_intensity(tuple_path_image):
    """
    Rescale image intensities, between the specified minima and maxima,
    by using a multiplicative factor.

    ----------
    minima, maxima : float
        Sets the range to which current intensities have to be rescaled.

    """
    path, img = tuple_path_image
    # the processing can be adjusted depending on the value of the path.
    print("process file:\n" + path)
    minima = 0
    maxima = 4095
    out_range = (minima, maxima)
    img_out = skimage.exposure.rescale_intensity(img, out_range=out_range)
    return img_out


params = Topology.create_default_params()

path_src = "../../image_samples/SurfTracking/Images"

params.images.path = path_src
params.images.path_ref = path_src
params.images.str_subset = ":4:2"
params.images.str_subset_ref = ":3"

params.surface_tracking.xmin = 200
params.surface_tracking.xmax = 250
params.surface_tracking.correct_pos = True

params.preproc.im2im = rescale_intensity

params.saving.how = "recompute"
# params.saving.path = str(self.path_out)
params.saving.postfix = "examples"

topology = Topology(params, logging_level="info")
# topology.make_code_graphviz('topo.dot')

topology.compute(executor="exec_sequential")
