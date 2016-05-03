from __future__ import print_function

import matplotlib.pyplot as plt
plt.ion()


class display(object):

    def __init__(self, im0, im1, piv_results=None):

        self.piv_results = piv_results

        if hasattr(piv_results, 'correls'):
            self.has_correls = True
        else:
            self.has_correls = False

        fig = plt.figure()
        fig.event_handler = self

        if self.has_correls:
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            self.ax2 = ax2
        else:
            ax1 = plt.gca()

        self.fig = fig
        self.ax1 = ax1

        self.axi0 = ax1.imshow(im0, interpolation='nearest')
        self.axi1 = ax1.imshow(im1, interpolation='nearest')
        self.axi1.set_visible(False)

        ax1.set_title('im 0 (alt+s to switch)')

        l, = ax1.plot(0, 0, 'oy')
        l.set_visible(False)
        t = fig.text(0.1, 0.05, '')

        self.t = t
        self.l = l

        ax1.set_xlim([0, im0.shape[1]])
        ax1.set_ylim([0, im0.shape[0]])
        ax1.set_xlabel('pixels')
        ax1.set_ylabel('pixels')

        if piv_results is not None:
            q = ax1.quiver(
                piv_results.xs, piv_results.ys,
                piv_results.deltaxs, piv_results.deltays,
                picker=10, color='w')

        self.q = q

        self.ind = 0
        fig.canvas.mpl_connect('pick_event', self.onpick)
        fig.canvas.mpl_connect('key_press_event', self.onclick)

        print('press alt+h for help')

        plt.show()

    def onclick(self, event):
        if event.key == 'alt+h':
            print('\nclick on a vector to show information\n'
                  'alt+s\t\t\t switch between images\n'
                  'alt+left or alt+right\t change vector.')

        if event.inaxes != self.ax1:
            return

        if event.key == 'alt+s':
            self.switch()

        if event.key == 'alt+left':
            self.select_arrow(self.ind - 1)

        if event.key == 'alt+right':
            self.select_arrow(self.ind + 1)

    def onpick(self, event):
        if event.artist != self.q:
            return True

        # the click locations
        # x = event.mouseevent.xdata
        # y = event.mouseevent.ydata
        ind = event.ind
        self.select_arrow(ind)

    def select_arrow(self, ind):
        ind = ind[0]
        q = self.q

        if ind >= len(q.X) or ind < 0:
            return

        self.ind = ind

        ix = q.X[ind]
        iy = q.Y[ind]
        self.l.set_visible(True)
        self.l.set_data(ix, iy)

        correl_max = self.piv_results.correls_max[ind]
        text = (
            'vector at ix = {} : iy = {} ; '
            'U = {:.3f} ; V = {:.3f}, C = {:.3f}').format(
                ix, iy, q.U[ind], q.V[ind], correl_max)

        if ind in self.piv_results.errors:
            text += ', error:' + self.piv_results.errors[ind]

        self.t.set_text(text)

        if self.has_correls:
            ax2 = self.ax2
            ax2.cla()
            alphac = self.piv_results.correls[ind]
            alphac_max = alphac.max()
            correl = correl_max/alphac_max * alphac

            ax2.imshow(correl, origin='lower', interpolation='none',
                       vmin=0, vmax=1)
            # ax2.pcolormesh(correl, vmin=0, vmax=1, shading='flat')

            ax2.plot(q.U[ind], q.V[ind], 'o')
            ax2.axis('scaled')
            ax2.set_xlim(-0.5, correl.shape[1]-0.5)
            ax2.set_ylim(-0.5, correl.shape[0]-0.5)
        self.fig.canvas.draw()

    def switch(self):
        self.axi0.set_visible(not self.axi0.get_visible())
        self.axi1.set_visible(not self.axi1.get_visible())

        self.ax1.set_title('im {} (alt+s to switch)'.format(
            int(self.axi1.get_visible())))

        self.fig.canvas.draw()
