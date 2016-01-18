
from __future__ import print_function

from scipy.ndimage import imread

import matplotlib.pyplot as plt
plt.ion()


def display(im0, im1, ixvec=None, iyvec=None, vecx=None, vecy=None):

    fig = plt.figure()
    ax = plt.gca()

    axi0 = ax.imshow(im0, interpolation='nearest')
    axi1 = ax.imshow(im1, interpolation='nearest')
    axi1.set_visible(False)

    ax.set_title('im 0 (alt+s to switch)')

    l, = ax.plot(0, 0, 'oy')
    l.set_visible(False)
    t = fig.text(0.1, 0.05, '')

    ax.set_xlim([0, im0.shape[1]])
    ax.set_ylim([0, im0.shape[0]])
    ax.set_xlabel('pixels')
    ax.set_ylabel('pixels')

    if ixvec is not None:
        q = ax.quiver(ixvec, iyvec, vecx, vecy, picker=10, color='w')

    def onclick(event):
        if event.inaxes != ax:
            return

        if event.key == 'alt+s':
            switch()

    def onpick(event):
        if event.artist != q:
            return True

        # the click locations
        # x = event.mouseevent.xdata
        # y = event.mouseevent.ydata

        ind = event.ind
        ix = q.X[ind]
        iy = q.Y[ind]
        l.set_visible(True)
        l.set_data(ix, iy)

        t.set_text('vector at ix = {} : iy = {} ; U = {} ; V = {}'.format(
            ix, iy, q.U[ind], q.V[ind]))
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', onclick)

    def switch():
        axi0.set_visible(not axi0.get_visible())
        axi1.set_visible(not axi1.get_visible())

        ax.set_title('im {} (alt+s to switch)'.format(int(axi1.get_visible())))

        fig.canvas.draw()

    plt.show()

if __name__ == '__main__':

    im0 = imread('samples/Karman/PIVlab_Karman_01.bmp')
    im1 = imread('samples/Karman/PIVlab_Karman_02.bmp')
    display(im0, im1)

