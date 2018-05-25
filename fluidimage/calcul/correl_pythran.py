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

    Notes
    -------

    im1_shape inf to im0_shape

    Returns
    -------

    the computing correlation (size of computed correlation = disp_max*2 + 1)

    """
    norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2))

    ny = nx = int(disp_max) * 2 + 1
    ny0, nx0 = im0.shape
    ny1, nx1 = im1.shape

    zero = np.float32(0.)
    correl = np.empty((ny, nx), dtype=np.float32)

    for xiy in range(disp_max + 1):
        dispy = -disp_max + xiy
        nymax = ny1 + min(ny0 // 2 - ny1 // 2 + dispy, 0)
        ny1dep = -min(ny0 // 2 - ny1 // 2 + dispy, 0)
        ny0dep = max(0, ny0 // 2 - ny1 // 2 + dispy)
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = nx1 + min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx1dep = -min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx0dep = max(0, nx0 // 2 - nx1 // 2 + dispx)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy, xix] = tmp / (nxmax * nymax)
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy, xix + disp_max + 1] = tmp / (nxmax * nymax)
    for xiy in range(disp_max):
        dispy = xiy + 1
        nymax = ny1 - max(ny0 // 2 + ny1 // 2 + dispy - ny0, 0)
        ny1dep = 0
        ny0dep = ny0 // 2 - ny1 // 2 + dispy
        for xix in range(disp_max + 1):
            dispx = -disp_max + xix
            nxmax = nx1 + min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx1dep = -min(nx0 // 2 - nx1 // 2 + dispx, 0)
            nx0dep = max(0, nx0 // 2 - nx1 // 2 + dispx)
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy + disp_max + 1, xix] = tmp / (nxmax * nymax)
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx
            tmp = zero
            for iy in range(nymax):
                for ix in range(nxmax):
                    tmp += (
                        im1[iy + ny1dep, ix + nx1dep]
                        * im0[ny0dep + iy, nx0dep + ix]
                    )
            correl[xiy + disp_max + 1, xix + disp_max + 1] = tmp / (nxmax * nymax)
    correl = correl * im1.size
    return correl, norm
