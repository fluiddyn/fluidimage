import numpy as np

# pythran export correl_pythran(float32[:,:], float32[:,:], float)


def correl_pythran(im0, im1, disp_max):
    """Correlations by hand using only numpy.

   Parameters
    ----------

    im0, im1 : images
      input images : 2D matrix

    disp_max : int
      displacement max.

    Returns
    -------

    the computing correlation (size of computed correlation = disp_max*2 + 1)

     """

    ny = disp_max * 2 + 1
    nx = ny
    ny0, nx0 = im0.shape
    ny1, nx1 = im1.shape

    correl = np.zeros((ny, nx), dtype=np.float32)

    for xiy in range(ny // 2 + 1):
        dispy = -disp_max + 2 * disp_max * xiy // (ny - 1)
        nymax = min(ny1 + dispy, ny0)
        for xix in range(nx // 2 + 1):
            dispx = -disp_max + 2 * disp_max * xix // (nx - 1)
            nxmax = min(nx1 + dispx, nx0)
            for iy in range(nymax):
                for ix in range(nxmax):
                    correl[xiy, xix] += im1[iy - dispy,
                                            ix - dispx] * im0[iy, ix]
        for xix in range(nx // 2):
            dispx = disp_max * (xix + 1) // (nx - 1)
            nxmax = min(nx0 - dispx, nx1)
            for iy in range(nymax):
                for ix in range(nxmax):
                    correl[xiy,
                           xix + disp_max + 1] += im1[iy - dispy,
                                                      ix] * im0[iy, ix + dispx]
    for xiy in range(ny // 2):
        dispy = disp_max * (xiy + 1) // (ny - 1)
        nymax = min(ny0 - dispy, ny1)
        for xix in range(nx // 2 + 1):
            dispx = -disp_max + 2 * disp_max * xix // (nx - 1)
            nxmax = min(nx1 + dispx, nx0)
            for iy in range(nymax):
                for ix in range(nxmax):
                    correl[xiy + disp_max + 1,
                           xix] += im1[iy, ix - dispx] * im0[iy + dispy, ix]
        for xix in range(nx // 2):
            dispx = disp_max * (xix + 1) // (nx - 1)
            nxmax = min(nx0 - dispx, nx1)
            for iy in range(nymax):
                for ix in range(nxmax):
                    correl[xiy + disp_max + 1,
                           xix + disp_max + 1] += im1[iy, ix] * im0[iy + dispy,
                                                                    ix + dispx]

    return correl
