from __future__ import print_function

import matplotlib.pyplot as plt
plt.ion()


class DisplayPIV(object):

    def __init__(self, im0, im1, piv_results=None, show_interp=False,
                 scale=0.2, show_error=True):

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
            if show_interp:
                if hasattr(piv_results, 'deltaxs_approx'):
                    deltaxs = piv_results.deltaxs_approx
                    deltays = piv_results.deltays_approx
                    xs = piv_results.ixvecs_approx
                    ys = piv_results.iyvecs_approx
                else:
                    deltaxs = piv_results.deltaxs_final
                    deltays = piv_results.deltays_final
                    xs = piv_results.ixvecs_final
                    ys = piv_results.iyvecs_final
            else:
                deltaxs = piv_results.deltaxs
                deltays = piv_results.deltays
                xs = piv_results.xs
                ys = piv_results.ys

            self.q = ax1.quiver(
                xs, ys,
                deltaxs, deltays,
                picker=20, color='w', scale_units='xy', scale=scale)

            self.inds_error = inds_error = piv_results.deltays_wrong.keys()

            if show_error:
                xs_wrong = xs[inds_error]
                ys_wrong = ys[inds_error]
                dxs_wrong = [piv_results.deltaxs_wrong[i] for i in inds_error]
                dys_wrong = [piv_results.deltays_wrong[i] for i in inds_error]
                self.q_wrong = ax1.quiver(
                    xs_wrong, ys_wrong,
                    dxs_wrong, dys_wrong,
                    picker=20, color='r', scale_units='xy', scale=scale)

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
        if not (event.artist == self.q or event.artist == self.q_wrong):
            return True

        # the click locations
        # x = event.mouseevent.xdata
        # y = event.mouseevent.ydata
        ind = event.ind
        self.select_arrow(ind, event.artist)

    def select_arrow(self, ind, artist=None):
        try:
            ind = ind[0]
        except (TypeError, IndexError):
            pass

        if artist is None:
            if ind in self.piv_results.errors.keys():
                artist = self.q_wrong
                ind = self.inds_error.index(ind)
            else:
                artist = self.q

        if artist == self.q:
            ind_all = ind
            q = self.q
        elif artist == self.q_wrong:
            ind_all = self.inds_error[ind]
            q = self.q_wrong

        if ind >= len(q.X) or ind < 0:
            return

        self.ind = ind_all

        ix = q.X[ind]
        iy = q.Y[ind]
        self.l.set_visible(True)
        self.l.set_data(ix, iy)

        correl_max = self.piv_results.correls_max[ind_all]
        text = (
            'vector at ix = {} : iy = {} ; '
            'U = {:.3f} ; V = {:.3f}, C = {:.3f}').format(
                ix, iy, q.U[ind], q.V[ind], correl_max)

        if ind_all in self.piv_results.errors:
            text += ', error: ' + self.piv_results.errors[ind_all]

        self.t.set_text(text)

        if self.has_correls:
            ax2 = self.ax2
            ax2.cla()
            alphac = self.piv_results.correls[ind_all]
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
