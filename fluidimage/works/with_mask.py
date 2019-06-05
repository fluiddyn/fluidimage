from . import BaseWork
from ..data_objects.piv import get_slices_from_strcrop


class BaseWorkWithMask(BaseWork):
    def _xymasked_from_xyoriginalimage(self, xs, ys):
        if self.params.mask.strcrop is not None:
            slices = get_slices_from_strcrop(self.params.mask.strcrop)
            if slices[1].start is not None:
                xs = xs - slices[1].start
            if slices[0].start is not None:
                ys = ys - slices[0].start
        return xs, ys

    def _xyoriginalimage_from_xymasked(self, xs, ys):
        if self.params.mask.strcrop is not None:
            slices = get_slices_from_strcrop(self.params.mask.strcrop)
            if slices[1].start is not None:
                xs = xs + slices[1].start
            if slices[0].start is not None:
                ys = ys + slices[0].start
        return xs, ys

    @classmethod
    def _complete_params_with_default_mask(cls, params):

        params._set_child("mask", attribs={"strcrop": None})

        params.mask._set_doc(
            """
Parameters describing how images are masked.

strcrop : None, str

    Two-dimensional slice (for example '100:600, :'). If None, the whole image
    is used.
"""
        )
