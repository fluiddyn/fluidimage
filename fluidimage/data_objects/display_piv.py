"""Display a PIV field (:mod:`fluidimage.data_objects.display_piv`)
===================================================================

.. autoclass:: DisplayPIV
   :members:
   :private-members:

"""

import matplotlib.pyplot as plt
import numpy as np

from ..calcul.correl import compute_indices_from_displacement


class DisplayPIV:
    """Display a piv result object."""

    def __init__(
        self,
        im0,
        im1,
        piv_results=None,
        show_interp=False,
        scale=0.2,
        show_error=True,
        pourcent_histo=99,
        hist=False,
        show_correl=True,
    ):

        self.piv_results = piv_results

        if (
            show_correl
            and hasattr(piv_results, "correls")
            and piv_results.correls
        ):
            self.show_correl = True
        else:
            self.show_correl = False

        fig = plt.figure()
        fig.event_handler = self

        if self.show_correl:
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)
            self.ax2 = ax2
        else:
            ax1 = plt.gca()

        self.fig = fig
        self.ax1 = ax1

        if im0 is not None:
            p0 = np.percentile(
                np.reshape(im0, (1, np.product(im0.shape))).transpose(),
                pourcent_histo,
            )
            p1 = np.percentile(
                np.reshape(im1, (1, np.product(im1.shape))).transpose(),
                pourcent_histo,
            )

            im0 = im0.copy()
            im1 = im1.copy()

            im0[im0 > p0] = p0
            im1[im1 > p1] = p1

            self.image0 = ax1.imshow(
                im0,
                interpolation="nearest",
                cmap=plt.cm.gray,
                origin="upper",
                extent=[0, im0.shape[1], im0.shape[0], 0],
                vmin=0,
                vmax=0.99 * im0.max(),
            )

            self.image1 = ax1.imshow(
                im1,
                interpolation="nearest",
                cmap=plt.cm.gray,
                origin="upper",
                extent=[0, im0.shape[1], im0.shape[0], 0],
                vmin=0,
                vmax=0.99 * im1.max(),
            )
            self.image1.set_visible(False)
        else:
            self.image0 = None

        l, = ax1.plot(0, 0, "oy")
        l.set_visible(False)

        ax1.set_title("im 0 (alt+s to switch)")

        t = fig.text(0.1, 0.05, "")

        self.t = t
        self.l = l

        if im0 is not None:
            ax1.set_xlim(0, im0.shape[1])
            ax1.set_ylim(im0.shape[0], 0)

        ax1.set_xlabel("pixels")
        ax1.set_ylabel("pixels")

        if piv_results is not None:
            if show_interp:
                if hasattr(piv_results, "deltaxs_approx"):
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

            if im0 is None:
                deltays *= -1

            self.q = ax1.quiver(
                xs,
                ys,
                deltaxs,
                -deltays,
                width=0.004,
                picker=10,
                color="c",
                scale_units="xy",
                scale=scale,
            )

            if show_error and not hasattr(piv_results, "deltays_wrong"):
                show_error = False

            if show_error:
                self.inds_error = inds_error = np.array(
                    list(piv_results.deltays_wrong.keys()), dtype=int
                )
                xs_wrong = xs[inds_error]
                ys_wrong = ys[inds_error]
                dxs_wrong = np.array(
                    [piv_results.deltaxs_wrong[i] for i in inds_error]
                )
                dys_wrong = np.array(
                    [piv_results.deltays_wrong[i] for i in inds_error]
                )

                if im0 is None:
                    dys_wrong *= -1

                self.q_wrong = ax1.quiver(
                    xs_wrong,
                    ys_wrong,
                    dxs_wrong,
                    -dys_wrong,
                    picker=10,
                    color="r",
                    scale_units="xy",
                    scale=scale,
                )

                inds_isnan = inds_error[np.isnan(dxs_wrong)]
                self.inds_isnan = inds_isnan
                xs_isnan = xs[inds_isnan]
                ys_isnan = ys[inds_isnan]

                zeros = np.zeros_like(xs_isnan)
                self.q_isnan = ax1.quiver(
                    xs_isnan,
                    ys_isnan,
                    zeros,
                    zeros,
                    minshaft=4,
                    picker=10,
                    color="r",
                    scale_units="xy",
                    scale=scale,
                )

        if hist:
            fig2, axes = plt.subplots(ncols=2)
            ax3, ax4 = axes.ravel()
            ind = (
                np.isnan(deltaxs)
                + np.isnan(deltays)
                + np.isinf(deltaxs)
                + np.isinf(deltays)
            )
            deltaxs2 = deltaxs[~ind]
            deltays2 = deltays[~ind]
            ax3.hist(deltaxs2, "fd", color="b", label=r"$\Delta x_s$")
            ax3.hist(deltays2, "fd", color="r", label=r"$\Delta y_s$")
            ax3.set_xlabel("displacement x (blue) and y (red) (pixels)")
            ax3.set_ylabel("histogram")
            ax3.legend()

            ax4.hist(piv_results.correls_max, "fd", color="g")
            ax4.set_xlabel("Maximum pixel correlation")
            ax4.set_ylabel("histogram")
            fig2.show()

        self.ind = 0
        fig.canvas.mpl_connect("pick_event", self.onpick)
        fig.canvas.mpl_connect("key_press_event", self.onclick)

        print("press alt+h for help")

        plt.show()

    def onclick(self, event):
        if event.key == "alt+h":
            print(
                "\nclick on a vector to show information\n"
                "alt+s\t\t\t switch between images\n"
                "alt+left or alt+right\t change vector."
            )

        if event.inaxes != self.ax1:
            return

        if event.key == "alt+s":
            self.switch()

        if event.key == "alt+left":
            self.select_arrow(self.ind - 1)

        if event.key == "alt+right":
            self.select_arrow(self.ind + 1)

    def onpick(self, event):
        if not (
            event.artist == self.q
            or event.artist == self.q_wrong
            or event.artist == self.q_isnan
        ):
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
            return

        if artist is None:
            print("artist is None")
            return

        # if ind in self.piv_results.errors.keys():
        #     artist = self.q_wrong
        #     ind = self.inds_error.index(ind)
        # else:
        #     artist = self.q

        if artist == self.q:
            ind_all = ind
            q = self.q
        elif artist == self.q_wrong:
            ind_all = self.inds_error[ind]
            q = self.q_wrong
        elif artist == self.q_isnan:
            ind_all = self.inds_isnan[ind]
            q = self.q_isnan
        else:
            raise NotImplementedError("other artist" + str(artist))

        if ind >= len(q.X) or ind < 0:
            return

        self.ind = ind_all

        result = self.piv_results

        ix = q.X[ind]
        iy = q.Y[ind]
        deltax = result.deltaxs[ind_all]
        deltay = result.deltays[ind_all]

        if np.isnan(deltax):
            deltax = result.deltaxs_wrong[ind_all]
            deltay = result.deltays_wrong[ind_all]

        self.l.set_visible(True)
        self.l.set_data(ix, iy)

        text = (
            f"vector at ix = {ix} : iy = {iy}"
            f" ; U = {deltax:.3f} ; V = {deltay:.3f}"
        )

        if result.correls_max is not None:
            correl_max = result.correls_max[ind_all]
            text += f", C = {correl_max:.3f}"

        if (
            self.piv_results.errors is not None
            and ind_all in self.piv_results.errors
        ):
            text += ", error: " + self.piv_results.errors[ind_all]

        self.t.set_text(text)

        if self.show_correl:
            ax2 = self.ax2
            ax2.cla()
            alphac = result.correls[ind_all]
            alphac_max = alphac.max()
            correl = correl_max / alphac_max * alphac

            ax2.imshow(correl, origin="lower", interpolation="none", vmin=0)

            ax2.plot(
                result.indices_no_displacement[0],
                result.indices_no_displacement[1],
                "or",
            )

            try:
                deltax -= result.deltaxs_approx0[ind_all]
                deltay -= result.deltays_approx0[ind_all]
            except AttributeError:
                pass

            i1, i0 = compute_indices_from_displacement(
                deltax, deltay, result.indices_no_displacement
            )

            ax2.plot(i1, i0, "xr")

            if hasattr(result, "deltaxs_final"):
                deltax = result.deltaxs_final[ind_all]
                deltay = result.deltays_final[ind_all]

                try:
                    deltax -= result.deltaxs_approx0[ind_all]
                    deltay -= result.deltays_approx0[ind_all]
                except AttributeError:
                    pass

                i1, i0 = compute_indices_from_displacement(
                    deltax, deltay, result.indices_no_displacement
                )

                ax2.plot(i1, i0, "ow")

            params = self.piv_results.params

            if params.piv0.nb_peaks_to_search > 1:
                other_peaks = result.secondary_peaks[ind_all]
                if other_peaks is not None:
                    if len(other_peaks) == 0:
                        s = "no other peak"
                        ax2.set_title(s)
                        print(s)
                    else:
                        s = "{} other peaks".format(len(other_peaks))
                        ax2.set_title(s)
                        print(s)

                    for (dx, dy, cmax) in other_peaks:
                        i1, i0 = compute_indices_from_displacement(
                            dx, dy, result.indices_no_displacement
                        )
                        ax2.plot(i1, i0, "sr")
                        print(dx, dy, cmax)

            if params.piv0.displacement_max is not None:
                circle = plt.Circle(
                    result.indices_no_displacement,
                    result.displacement_max,
                    color="b",
                    fill=False,
                )
                ax2.add_artist(circle)

            ax2.axis("scaled")
            ax2.set_xlim(-0.5, correl.shape[1] - 0.5)
            ax2.set_ylim(-0.5, correl.shape[0] - 0.5)
        self.fig.canvas.draw()

    def switch(self):
        if self.image0 is not None:
            self.image0.set_visible(not self.image0.get_visible())
            self.image1.set_visible(not self.image1.get_visible())

            self.ax1.set_title(
                "im {} (alt+s to switch)".format(int(self.image1.get_visible()))
            )

            self.fig.canvas.draw()
