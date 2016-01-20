
import display2


class HeavyPIVResults(object):
    def __init__(self, deltaxs, deltays, correls, couple, work):
        self.deltaxs = deltaxs
        self.deltays = deltays
        self.correls = correls
        self.couple = couple
        self.work = work

    def get_images(self):
        return self.couple.serie.get_arrays()

    def display(self):
        im0, im1 = self.couple.get_arrays()
        display2.display2(
            im0, im1, self.work.inds_x_vec, self.work.inds_y_vec,
            self.deltaxs, self.deltays,self.correls)
