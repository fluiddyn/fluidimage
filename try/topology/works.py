
from data_objects import ArrayCouple, HeavyPIVResult


def load_image(path):
    with open(path) as f:
        return f.readline().strip()


class PIVWork(object):
    def calcul(self, couple):
        if not isinstance(couple, ArrayCouple):
            raise ValueError

        s = self.poum()
        print(s)

        a0, a1 = couple.arrays

        return HeavyPIVResult(a0, a1)

    def poum(self):
        return 'poum'
