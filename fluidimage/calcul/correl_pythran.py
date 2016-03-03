import numpy as np


# pythran export correl_pythran(float32[][], float32[][], int)

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
    norm = np.sum(im1**2) * im0.size

    ny = nx = int(disp_max) * 2 + 1
    ny0, nx0 = im0.shape
    ny1, nx1 = im1.shape

    zero = np.float32(0.)
    correl = np.empty((ny, nx), dtype=np.float32)

    for xiy in range(disp_max + 1):
        dispy = -disp_max + xiy
        nymax = min(ny1 + dispy, ny0)
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = min(nx1 + dispx, nx0)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += im1[iy - dispy, ix - dispx] * im0[iy, ix]
            correl[xiy, xix] = tmp
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = min(nx0 - dispx, nx1)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += im1[iy - dispy, ix] * im0[iy, ix + dispx]
            correl[xiy, xix + disp_max + 1] = tmp
    for xiy in range(disp_max):
        dispy = xiy + 1
        nymax = min(ny0 - dispy, ny1)
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = min(nx1 + dispx, nx0)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += im1[iy, ix - dispx] * im0[iy + dispy, ix]
            correl[xiy + disp_max + 1, xix] = tmp
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = min(nx0 - dispx, nx1)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += im1[iy, ix] * im0[iy + dispy, ix + dispx]
            correl[xiy + disp_max + 1, xix + disp_max + 1] = tmp

    return correl, norm
