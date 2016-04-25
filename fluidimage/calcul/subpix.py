"""Find subpixel peak
=====================

.. autoclass:: SubPix
   :members:
   :private-members:

"""

import numpy as np

from.errors import PIVError


class SubPix(object):
    """Subpixel finder

    .. todo::

       - evaluate subpix methods.

       - same subpix methods as in UVmat...

    """
    methods = ['2d_gaussian', 'centroid', 'no_subpix']

    def __init__(self, method='centroid'):
        self.method = method

        n = self.n = 2
        xs = ys = np.arange(-n, n+1, dtype=float)
        X, Y = np.meshgrid(xs, ys)

        # init for centroid method
        self.X_centroid = X
        self.Y_centroid = Y

        # init for 2d_gaussian method
        nx, ny = X.shape
        X = X.ravel()
        Y = Y.ravel()
        M = np.reshape(np.concatenate(
            (X**2, Y**2, X, Y, np.ones(nx*ny))), (5, nx*ny)).T
        self.Minv_subpix = np.linalg.pinv(M)

    def compute_subpix(self, correl, ix, iy, method=None):
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

        if method not in self.methods:
            raise ValueError('method has to be in {}'.format(self.methods))

        n = self.n

        ny, nx = correl.shape

        if iy-n < 0 or iy+n+1 > ny or \
           ix-n < 0 or ix+n+1 > nx:
            raise PIVError(explanation='close boundary',
                           result_compute_subpix=(iy, ix))

        if method == '2d_gaussian':

            correl = correl[iy-n:iy+n+1, ix-n:ix+n+1]
            ny, nx = correl.shape

            assert nx == ny == 2*n + 1

            correl_map = correl.ravel()
            correl_map[correl_map == 0.] = 1e-6

            coef = np.dot(self.Minv_subpix, np.log(correl_map))

            sigmax = 1/np.sqrt(-2*coef[0])
            sigmay = 1/np.sqrt(-2*coef[1])
            deplx = coef[2]*sigmax**2
            deply = coef[3]*sigmay**2

        elif method == 'centroid':
            n = self.n

            correl = correl[iy-n:iy+n+1, ix-n:ix+n+1]
            ny, nx = correl.shape

            sum_correl = np.sum(correl)

            deplx = np.sum(self.X_centroid * correl) / sum_correl
            deply = np.sum(self.Y_centroid * correl) / sum_correl

        elif method == 'no_subpix':
            deplx = deply = 0.

        # print('deplxy', deplx, deply, iy, ix)

        if abs(deplx) > 1 or abs(deply) > 1:
            raise PIVError(explanation='wrong subpix',
                           result_compute_subpix=(iy, ix))

        return deplx + ix, deply + iy
