

class PIVError(Exception):
    """No peak"""
    explanation = 'no peak'

    def __init__(self, *args, **kargs):
        for k, v in kargs.items():
            self.__dict__[k] = v
        super(PIVError, self).__init__(*args)
