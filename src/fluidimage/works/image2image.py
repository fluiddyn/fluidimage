"""Image to image processing

Provides

.. autoclass:: WorkImage2Image
   :members:
   :private-members:

"""

from fluidimage.data_objects.display_pre import DisplayPreProc
from fluidimage.image2image import (
    complete_im2im_params_with_default,
    init_im2im_function,
)
from fluidimage.works import BaseWorkFromImage


class WorkImage2Image(BaseWorkFromImage):

    @classmethod
    def _complete_params_with_default(cls, params):
        super()._complete_params_with_default(params)
        complete_im2im_params_with_default(params)

    def __init__(self, params):
        super().__init__(params)
        self._init_serie()

        self.im2im_obj, self.im2im_func = init_im2im_function(
            im2im=params.im2im, args_init=params.args_init
        )

    def calcul(self, tuple_image_name):
        return self.im2im_func(tuple_image_name)

    def display(self, ind=0, hist=False):
        """Display figures to study the preprocessing"""

        tuple_image_name0 = self.get_image_name(ind)
        tuple_image_name1 = self.get_image_name(ind + 1)

        result0 = self.calcul(tuple_image_name0)
        result1 = self.calcul(tuple_image_name1)

        arr_input0, _ = tuple_image_name0
        arr_input1, _ = tuple_image_name1

        arr_output0, _ = result0
        arr_output1, _ = result1

        return DisplayPreProc(
            arr_input0, arr_input1, arr_output0, arr_output1, hist=hist
        )


Work = WorkImage2Image
