import numpy as np

# pythran export correl_pythran(float32[:,:], float32[:,:], float)


def correl_pythran(im0, im1, disp_max):
    """Correlations by hand using only numpy.

    disp_max => filter image size (im0_crop)

     .. math::
     """
    # correl (\xi_x, \xi_y) = \int_{im1 inter im0} im1*im0_{\delta \xi}

    ny = disp_max * 2 + 1
    nx = ny
    ny0, nx0 = im1.shape
    ny1, nx1 = im0.shape

#  if mode == 'same':

    correl = np.zeros((ny, nx), dtype=np.float32)

    for xix in range(nx // 2 + 1):
        dispx = -disp_max + 2 * disp_max * xix // (nx - 1)
        for xiy in range(ny // 2 + 1):
            dispy = -disp_max + 2 * disp_max * xiy // (ny - 1)
            for ix in range(nx0 + dispx):
                for iy in range(ny0 + dispy):
                    correl[xiy, xix] += im1[iy - dispy,
                                            ix - dispx] * im0[iy, ix]
        for xiy in range(ny // 2):
            dispy = disp_max * (xiy + 1) // (ny - 1)
            for ix in range(nx0 + dispx):
                for iy in range(ny0 - dispy):
                    correl[xiy + disp_max + 1,
                           xix] += im1[iy, ix - dispx] * im0[iy + dispy, ix]
    for xix in range(nx // 2):
        dispx = disp_max * (xix + 1) // (nx - 1)
        for xiy in range(ny // 2 + 1):
            dispy = -disp_max + 2 * disp_max * xiy // (ny - 1)
            for ix in range(nx0 - dispx):
                for iy in range(ny0 + dispy):
                    correl[xiy,
                           xix + disp_max + 1] += im1[iy - dispy,
                                                      ix] * im0[iy, ix + dispx]
        for xiy in range(ny // 2):
            dispy = disp_max * (xiy + 1) // (ny - 1)
            for ix in range(nx0 - dispx):
                for iy in range(ny0 - dispy):
                    correl[xiy + disp_max + 1,
                           xix + disp_max + 1] += im1[iy, ix] * im0[iy + dispy,
                                                                    ix + dispx]
# else TODO mode='valid'

    return correl
