import os
import numpy as np

try:
    import pycuda._driver
except ImportError:
    pass
else:
    try:
        import pycuda.autoinit
        import pycuda.compiler
        import pycuda.gpuarray
        import pycuda.driver
    except (ImportError, pycuda._driver.RuntimeError):
        pass


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def correl_pycuda(im0, im1, disp_max):
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
    norm = np.sqrt(np.sum(im1 ** 2) * np.sum(im0 ** 2)) * im0.size

    ny = nx = int(disp_max) * 2 + 1
    ny0, nx0 = im0.shape
    ny1, nx1 = im1.shape

    # zero = np.float32(0.)
    correl = np.empty((ny, nx), dtype=np.float32)

    # Load the kernel and compile it.
    kernel_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "correl.cu"
    )
    f = open(kernel_file, "r")
    kernel = pycuda.compiler.SourceModule(f.read())
    f.close()
    correlate_cuda = kernel.get_function("cucorrelate")
    # correlate_cuda.prepare([np.intp, np.intp, np.intp, np.int32,
    #                        np.int32,np.int32,np.int32,np.int32,np.int32,
    #                        np.int32,np.int32,np.int32,np.int32])

    #    correlate_cuda.prepare([numpy.intp,numpy.intp,numpy.intp,numpy.int32,numpy.int32,
    #                    numpy.int32,numpy.int32,numpy.int32,numpy.int32,
    #                    numpy.int32,numpy.int32,numpy.int32,numpy.int32],block=(nthreads_x,nthreads_x,nthreads_x))
    kernel_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "reduction_kernel.cu"
    )
    f = open(kernel_file, "r")
    kernel1 = pycuda.compiler.SourceModule(f.read())
    f.close()

    reduction_cuda = kernel1.get_function("reduce1")

    # CUDA parameters that seem to work well. The number of threads per tile
    # (the tile_size) should be a power of 2 for the parallel reduction to
    # work right!
    threads_per_block = 16
    blocks_per_tile = max(nx0, ny0) / threads_per_block

    maxthreads = 16

    g_im0data = pycuda.gpuarray.to_gpu(im0)
    g_im1data = pycuda.gpuarray.to_gpu(im1)

    g_odata = pycuda.gpuarray.empty(nx0 * ny0, np.float32)
    g_sumdata = pycuda.gpuarray.empty(1, np.float32)

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

            threads_reduce = maxthreads
            blocks_reduce = nx0 * ny0
            if blocks_reduce < maxthreads:
                threads_reduce = nextpow2(blocks_reduce)
            blocks_reduce = (blocks_reduce + threads_reduce - 1) // threads_reduce
            correlate_cuda(
                g_im0data.gpudata,
                g_im1data.gpudata,
                g_odata.gpudata,
                np.int32(nx0),
                np.int32(nx1),
                np.int32(ny0),
                np.int32(dispx),
                np.int32(dispy),
                np.int32(nymax),
                np.int32(ny1dep),
                np.int32(ny0dep),
                np.int32(nx0dep),
                np.int32(nxmax),
                np.int32(nx1dep),
                block=(threads_per_block, threads_per_block, 1),
                grid=(blocks_per_tile, blocks_per_tile),
            )
            while blocks_reduce != 1:
                reduction_cuda(
                    g_odata.gpudata,
                    g_odata.gpudata,
                    block=(threads_reduce, 1, 1),
                    grid=(blocks_reduce, 1),
                    shared=4 * threads_reduce,
                )
                if blocks_reduce < maxthreads:
                    threads_reduce = nextpow2(blocks_reduce)
                blocks_reduce = (
                    blocks_reduce + threads_reduce - 1
                ) // threads_reduce
            reduction_cuda(
                g_odata.gpudata,
                g_sumdata.gpudata,
                block=(threads_reduce, 1, 1),
                grid=(blocks_reduce, 1),
                shared=4 * threads_reduce,
            )

            correl[xiy, xix] = g_sumdata.get()

        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx

            threads_reduce = maxthreads
            blocks_reduce = nx0 * ny0
            if blocks_reduce < maxthreads:
                threads_reduce = nextpow2(blocks_reduce)
            blocks_reduce = (blocks_reduce + threads_reduce - 1) // threads_reduce
            correlate_cuda(
                g_im0data.gpudata,
                g_im1data.gpudata,
                g_odata.gpudata,
                np.int32(nx0),
                np.int32(nx1),
                np.int32(ny0),
                np.int32(dispx),
                np.int32(dispy),
                np.int32(nymax),
                np.int32(ny1dep),
                np.int32(ny0dep),
                np.int32(nx0dep),
                np.int32(nxmax),
                np.int32(nx1dep),
                block=(threads_per_block, threads_per_block, 1),
                grid=(blocks_per_tile, blocks_per_tile),
            )
            while blocks_reduce != 1:
                reduction_cuda(
                    g_odata.gpudata,
                    g_odata.gpudata,
                    block=(threads_reduce, 1, 1),
                    grid=(blocks_reduce, 1),
                    shared=4 * threads_reduce,
                )
                if blocks_reduce < maxthreads:
                    threads_reduce = nextpow2(blocks_reduce)
                blocks_reduce = (
                    blocks_reduce + threads_reduce - 1
                ) // threads_reduce
            reduction_cuda(
                g_odata.gpudata,
                g_sumdata.gpudata,
                block=(threads_reduce, 1, 1),
                grid=(blocks_reduce, 1),
                shared=4 * threads_reduce,
            )

            correl[xiy, xix + disp_max + 1] = g_sumdata.get()

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

            threads_reduce = maxthreads
            blocks_reduce = nx0 * ny0
            if blocks_reduce < maxthreads:
                threads_reduce = nextpow2(blocks_reduce)
            blocks_reduce = (blocks_reduce + threads_reduce - 1) // threads_reduce
            correlate_cuda(
                g_im0data.gpudata,
                g_im1data.gpudata,
                g_odata.gpudata,
                np.int32(nx0),
                np.int32(nx1),
                np.int32(ny0),
                np.int32(dispx),
                np.int32(dispy),
                np.int32(nymax),
                np.int32(ny1dep),
                np.int32(ny0dep),
                np.int32(nx0dep),
                np.int32(nxmax),
                np.int32(nx1dep),
                block=(threads_per_block, threads_per_block, 1),
                grid=(blocks_per_tile, blocks_per_tile),
            )
            while blocks_reduce != 1:
                reduction_cuda(
                    g_odata.gpudata,
                    g_odata.gpudata,
                    block=(threads_reduce, 1, 1),
                    grid=(blocks_reduce, 1),
                    shared=4 * threads_reduce,
                )
                if blocks_reduce < maxthreads:
                    threads_reduce = nextpow2(blocks_reduce)
                blocks_reduce = (
                    blocks_reduce + threads_reduce - 1
                ) // threads_reduce
            reduction_cuda(
                g_odata.gpudata,
                g_sumdata.gpudata,
                block=(threads_reduce, 1, 1),
                grid=(blocks_reduce, 1),
                shared=4 * threads_reduce,
            )

            correl[xiy + disp_max + 1, xix] = g_sumdata.get()
        for xix in range(disp_max):
            dispx = xix + 1
            nxmax = nx1 - max(nx0 // 2 + nx1 // 2 + dispx - nx0, 0)
            nx1dep = 0
            nx0dep = nx0 // 2 - nx1 // 2 + dispx

            threads_reduce = maxthreads
            blocks_reduce = nx0 * ny0
            if blocks_reduce < maxthreads:
                threads_reduce = nextpow2(blocks_reduce)
            blocks_reduce = (blocks_reduce + threads_reduce - 1) // threads_reduce
            correlate_cuda(
                g_im0data.gpudata,
                g_im1data.gpudata,
                g_odata.gpudata,
                np.int32(nx0),
                np.int32(nx1),
                np.int32(ny0),
                np.int32(dispx),
                np.int32(dispy),
                np.int32(nymax),
                np.int32(ny1dep),
                np.int32(ny0dep),
                np.int32(nx0dep),
                np.int32(nxmax),
                np.int32(nx1dep),
                block=(threads_per_block, threads_per_block, 1),
                grid=(blocks_per_tile, blocks_per_tile),
            )
            while blocks_reduce != 1:
                reduction_cuda(
                    g_odata.gpudata,
                    g_odata.gpudata,
                    block=(threads_reduce, 1, 1),
                    grid=(blocks_reduce, 1),
                    shared=4 * threads_reduce,
                )
                if blocks_reduce < maxthreads:
                    threads_reduce = nextpow2(blocks_reduce)
                blocks_reduce = (
                    blocks_reduce + threads_reduce - 1
                ) // threads_reduce
            reduction_cuda(
                g_odata.gpudata,
                g_sumdata.gpudata,
                block=(threads_reduce, 1, 1),
                grid=(blocks_reduce, 1),
                shared=4 * threads_reduce,
            )

            correl[xiy + disp_max + 1, xix + disp_max + 1] = g_sumdata.get()

    return correl, norm
