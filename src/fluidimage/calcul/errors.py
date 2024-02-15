"""Errors (:mod:`fluidimage.calcul.errors`)
===========================================


"""


class PIVError(Exception):
    """No peak"""

    def __init__(self, *args, **kargs):
        self.explanation = "General no peak error"
        for k, v in kargs.items():
            self.__dict__[k] = v
        super().__init__(*args)
