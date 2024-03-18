""""Image to image" processing
==============================

.. autofunction:: apply_im2im_filter

.. autofunction:: im2im_func_example

.. autoclass:: Im2ImExample
   :members:

.. autoclass:: Work
   :members:
   :private-members:

.. autoclass:: Topology
   :members:
   :private-members:

"""

from pathlib import Path

import numpy as np

from fluiddyn.util.serieofarrays import SerieOfArraysFromFiles
from fluidimage.topologies.image2image import Topology, TopologyImage2Image
from fluidimage.works.image2image import (
    Work,
    WorkImage2Image,
    init_im2im_function,
)


def im2im_func_example(tuple_image_path):
    """Process one image

    This is just an example to show how to write functions which can be plugged
    to the class
    :class:`fluidimage.topologies.image2image.TopologyImage2Image`.

    """
    path, image = tuple_image_path
    # the processing can be adjusted depending on the value of the path.
    print("process file:\n" + path)
    image_out = np.round(image * (255 / image.max())).astype(np.uint8)
    return image_out


class Im2ImExample:
    """Process one image

    This is just an example to show how to write classes which can be plugged
    to the class
    :class:`fluidimage.topologies.image2image.TopologyImage2Image`.

    """

    def __init__(self, arg0, arg1):
        print("init with arguments:", arg0, arg1)
        self.arg0 = arg0
        self.arg1 = arg1

    # time consuming tasks can be done here

    def calcul(self, tuple_image_path):
        """Method processing one image"""
        print(
            "calcul with arguments (unused in the example):", self.arg0, self.arg1
        )
        return im2im_func_example(tuple_image_path)


def apply_im2im_filter(serie, im2im=None, args_init=()):
    """Apply an image-to-image filter to a serie of images.

    Parameters
    ----------

    serie : :class:`fluiddyn.util.serieofarrays.SerieOfArraysFromFiles`

    im2im : Optional[str]

    args_init : tuple

    """
    if im2im is None:
        return serie

    _, im2im_func = init_im2im_function(im2im, args_init)

    if not isinstance(serie, SerieOfArraysFromFiles):
        raise NotImplementedError

    result = {"names": [], "arrays": [], "paths": []}

    arrays = serie.get_arrays()
    paths = serie.get_path_arrays()

    for arr, path in zip(arrays, paths):
        new_arr = im2im_func((path, arr))
        result["arrays"].append(new_arr)
        result["paths"].append(path)
        result["names"].append(Path(path).name)

    return result


__all__ = ["WorkImage2Image", "Work", "TopologyImage2Image", "Topology"]
