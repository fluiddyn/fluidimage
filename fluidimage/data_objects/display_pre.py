"""Display pairs of images (:mod:`fluidimage.data_objects.display_pre`)
=======================================================================

.. autoclass:: DisplayPreProc
   :members:
   :private-members:

"""

import matplotlib.pyplot as plt
import numpy as np


class DisplayPreProc:
    """Display pairs of images"""

    def __init__(self, im0, im1, im0p, im1p, pourcent_histo=99, hist=False):

        fig = plt.figure()
        fig.event_handler = self

        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

        p0 = np.percentile(
            np.reshape(im0, (1, np.product(im0.shape))).transpose(),
            pourcent_histo,
        )
        p1 = np.percentile(
            np.reshape(im1, (1, np.product(im1.shape))).transpose(),
            pourcent_histo,
        )
        p0p = np.percentile(
            np.reshape(im0p, (1, np.product(im0p.shape))).transpose(),
            pourcent_histo,
        )
        p1p = np.percentile(
            np.reshape(im1p, (1, np.product(im1p.shape))).transpose(),
            pourcent_histo,
        )
        im0[im0 > p0] = p0
        im1[im1 > p1] = p1
        im0p[im0p > p0p] = p0p
        im1p[im1p > p1p] = p1p

        self.image0 = ax1.imshow(
            im0,
            interpolation="nearest",
            cmap=plt.cm.gray,
            origin="upper",
            extent=[0, im0.shape[1], im0.shape[0], 0],
        )

        self.image1 = ax1.imshow(
            im1,
            interpolation="nearest",
            cmap=plt.cm.gray,
            origin="upper",
            extent=[0, im0.shape[1], im0.shape[0], 0],
        )
        self.image1.set_visible(False)

        self.image0p = ax2.imshow(
            im0p,
            interpolation="nearest",
            cmap=plt.cm.gray,
            origin="upper",
            extent=[0, im0p.shape[1], im0p.shape[0], 0],
        )

        self.image1p = ax2.imshow(
            im1p,
            interpolation="nearest",
            cmap=plt.cm.gray,
            origin="upper",
            extent=[0, im0p.shape[1], im0p.shape[0], 0],
        )
        self.image1p.set_visible(False)

        if hist:
            fig2 = plt.figure()
            ax3 = plt.subplot(121)
            ax4 = plt.subplot(122)
            hist0 = np.histogram(
                np.reshape(im0, (1, np.product(im0.shape))).transpose(), bins="fd"
            )
            hist1 = np.histogram(
                np.reshape(im1, (1, np.product(im1.shape))).transpose(), bins="fd"
            )
            hist0p = np.histogram(
                np.reshape(im0p, (1, np.product(im0p.shape))).transpose(),
                bins="fd",
            )
            hist1p = np.histogram(
                np.reshape(im1p, (1, np.product(im1p.shape))).transpose(),
                bins="fd",
            )
            incr = 1
            ax3.plot(hist0[1][0:-1:incr], hist0[0][0::incr], "k+")
            ax3.plot(hist0p[1][0:-1:incr], hist0p[0][0::incr], "r+")
            ax4.plot(hist1[1][0:-1:incr], hist1[0][0::incr], "k+")
            ax4.plot(hist1p[1][0:-1:incr], hist1p[0][0::incr], "r+")

            ax3.set_xlim(-10, max([p0, p0p]))
            ax4.set_xlim(-10, max([p1, p1p]))
            ax3.set_ylim(0, max(hist0[0]))
            ax4.set_ylim(0, max(hist1[0]))
            fig2.show()

        l, = ax1.plot(0, 0, "oy")
        l.set_visible(False)

        ax1.set_title("im 0 (alt+s to switch)")

        ax1.set_xlim(0, im0.shape[1])
        ax1.set_ylim(im0.shape[0], 0)
        ax1.set_xlabel("pixels")
        ax1.set_ylabel("pixels")

        ax1.set_xlim(0, im0.shape[1])
        ax1.set_ylim(im0.shape[0], 0)

        ax1.set_xlabel("pixels")
        ax1.set_ylabel("pixels")

        l, = ax2.plot(0, 0, "oy")
        l.set_visible(False)

        ax2.set_title("im 0p (alt+s to switch)")

        ax2.set_xlim(0, im0p.shape[1])
        ax2.set_ylim(im0p.shape[0], 0)
        ax2.set_xlabel("pixels")
        ax2.set_ylabel("pixels")

        self.ind = 0
        fig.canvas.mpl_connect("key_press_event", self.onclick)
        print("press alt+h for help")

        plt.show()

    def onclick(self, event):
        if event.key == "alt+s":
            self.switch()

    def switch(self):
        self.image0.set_visible(not self.image0.get_visible())
        self.image1.set_visible(not self.image1.get_visible())

        self.ax1.set_title(
            "im {} (alt+s to switch)".format(int(self.image1.get_visible()))
        )

        self.fig.canvas.draw()

        self.image0p.set_visible(not self.image0p.get_visible())
        self.image1p.set_visible(not self.image1p.get_visible())

        self.ax2.set_title(
            "im {}p (alt+s to switch)".format(int(self.image1p.get_visible()))
        )
        self.fig.canvas.draw()
