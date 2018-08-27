"""Find subpixel peak
=====================

.. autoclass:: SubPix
   :members:
   :private-members:

"""

import numpy as np

from .errors import PIVError

from .subpix_pythran import compute_subpix_2d_gaussian2


class SubPix(object):
    """Subpixel finder

    .. todo::

       - evaluate subpix methods.

    """

    methods = ["2d_gaussian", "2d_gaussian2", "centroid", "no_subpix"]

    def __init__(self, method="centroid", nsubpix=None):
        if method == "2d_gaussian2" and nsubpix is not None:
            raise ValueError(
                "Subpixel method '2d_gaussian2' doesn't require nsubpix. "
                "In this case, nsubpix has to be equal to None."
            )

        self.prepare_subpix(method, nsubpix)

    def prepare_subpix(self, method, nsubpix):
        self.method = method
        if nsubpix is None:
            nsubpix = 1
        self.n = nsubpix
        xs = ys = np.arange(-nsubpix, nsubpix + 1, dtype=float)
        X, Y = np.meshgrid(xs, ys)

        # init for centroid method
        self.X_centroid = X
        self.Y_centroid = Y

        # init for 2d_gaussian method
        nx, ny = X.shape
        X = X.ravel()
        Y = Y.ravel()
        M = np.reshape(
            np.concatenate((X ** 2, Y ** 2, X, Y, np.ones(nx * ny))), (5, nx * ny)
        ).T
        self.Minv_subpix = np.linalg.pinv(M)

    def compute_subpix(self, correl, ix, iy, method=None, nsubpix=None):
        """Find peak

        Parameters
        ----------

        correl: numpy.ndarray

          Normalized correlation

        ix: integer

        iy: integer

        method: str {'centroid', '2d_gaussian'}

        Notes
        -----

        The two methods...

        using linalg.solve (buggy?)

        """
        if method is None:
            method = self.method
        if nsubpix is None:
            nsubpix = self.n

        if method != self.method or nsubpix != self.n:
            if method is None:
                method = self.method
            if nsubpix is None:
                nsubpix = self.n
            self.prepare_subpix(method, nsubpix)

        if method not in self.methods:
            raise ValueError("method has to be in {}".format(self.methods))

        ny, nx = correl.shape

        if (
            iy - nsubpix < 0
            or iy + nsubpix + 1 > ny
            or ix - nsubpix < 0
            or ix + nsubpix + 1 > nx
        ):
            raise PIVError(
                explanation="close boundary", result_compute_subpix=(iy, ix)
            )

        if method == "2d_gaussian":

            correl_crop = correl[
                iy - nsubpix : iy + nsubpix + 1, ix - nsubpix : ix + nsubpix + 1
            ]
            ny, nx = correl_crop.shape

            assert nx == ny == 2 * nsubpix + 1

            correl_map = correl_crop.ravel()
            correl_map[correl_map <= 0.] = 1e-6

            coef = np.dot(self.Minv_subpix, np.log(correl_map))
            if coef[0] > 0 or coef[1] > 0:
                return self.compute_subpix(
                    correl, ix, iy, method="centroid", nsubpix=nsubpix
                )

            sigmax = 1 / np.sqrt(-2 * coef[0])
            sigmay = 1 / np.sqrt(-2 * coef[1])
            deplx = coef[2] * sigmax ** 2
            deply = coef[3] * sigmay ** 2

            if np.isnan(deplx) or np.isnan(deply):
                return self.compute_subpix(
                    correl, ix, iy, method="centroid", nsubpix=nsubpix
                )

        if method == "2d_gaussian2":
            deplx, deply, correl_crop = compute_subpix_2d_gaussian2(
                correl, int(ix), int(iy)
            )

        elif method == "centroid":

            correl_crop = correl[
                iy - nsubpix : iy + nsubpix + 1, ix - nsubpix : ix + nsubpix + 1
            ]
            ny, nx = correl_crop.shape

            sum_correl = np.sum(correl_crop)

            deplx = np.sum(self.X_centroid * correl_crop) / sum_correl
            deply = np.sum(self.Y_centroid * correl_crop) / sum_correl

        elif method == "no_subpix":
            deplx = deply = 0.

        if deplx ** 2 + deply ** 2 > 2 * (0.5 + nsubpix) ** 2:
            if method == "2d_gaussian2":
                return self.compute_subpix(
                    correl, ix, iy, method="centroid", nsubpix=nsubpix
                )
            print(
                (
                    "Wrong subpix for one vector:"
                    " deplx**2 + deply**2 > (0.5+nsubpix)**2\n"
                    "method: " + method + "\ndeplx, deply = ({}, {})\n"
                    "correl_subpix =\n{}"
                ).format(deplx, deply, correl_crop)
            )
            raise PIVError(
                explanation="wrong subpix", result_compute_subpix=(iy, ix)
            )

        return deplx + ix, deply + iy
