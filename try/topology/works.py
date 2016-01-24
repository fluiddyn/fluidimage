
from data_objects import ArrayCouple, HeavyPIVResult

# try:
#     from scipy.ndimage import imread
# except ImportError:
#     from scipy.misc import imread


def load_image(path):
    with open(path) as f:
        return f.readline().strip()


class PIVWork(object):
    def calcul(self, couple):
        if not isinstance(couple, ArrayCouple):
            raise ValueError

        a0, a1 = couple.arrays

        return HeavyPIVResult(a0, a1)
