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
