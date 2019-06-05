import ctypes
import time

import numpy as np

import pycuda.autoinit
import pycuda.gpuarray as gpuarray

CUSOLVER_STATUS_SUCCESS = 0

libcusolver = ctypes.cdll.LoadLibrary("libcusolver.so")

libcusolver.cusolverDnCreate.restype = int
libcusolver.cusolverDnCreate.argtypes = [ctypes.c_void_p]


def cusolverDnCreate():
    handle = ctypes.c_void_p()
    status = libcusolver.cusolverDnCreate(ctypes.byref(handle))
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError("error!")

    return handle.value


libcusolver.cusolverDnDestroy.restype = int
libcusolver.cusolverDnDestroy.argtypes = [ctypes.c_void_p]


def cusolverDnDestroy(handle):
    status = libcusolver.cusolverDnDestroy(handle)
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError("error!")


libcusolver.cusolverDnSgetrf_bufferSize.restype = int
libcusolver.cusolverDnSgetrf_bufferSize.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork):
    status = libcusolver.cusolverDnSgetrf_bufferSize(
        handle, m, n, int(A.gpudata), n, ctypes.pointer(Lwork)
    )
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError("error!")


libcusolver.cusolverDnSgetrf.restype = int
libcusolver.cusolverDnSgetrf.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]


def cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo):
    status = libcusolver.cusolverDnSgetrf(
        handle,
        m,
        n,
        int(A.gpudata),
        lda,
        int(Workspace.gpudata),
        int(devIpiv.gpudata),
        int(devInfo.gpudata),
    )
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError("error!")


libcusolver.cusolverDnSgetrs.restype = int
libcusolver.cusolverDnSgetrs.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
]


def cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo):
    status = libcusolver.cusolverDnSgetrs(
        handle,
        trans,
        n,
        nrhs,
        int(A.gpudata),
        lda,
        int(devIpiv.gpudata),
        int(B.gpudata),
        ldb,
        int(devInfo.gpudata),
    )
    if status != CUSOLVER_STATUS_SUCCESS:
        raise RuntimeError("error!")


if __name__ == "__main__":
    import numpy as np

    m = 6400
    n = 6400
    a = np.asarray(np.random.rand(m, n), np.float32)
    init_time = time.time()
    a_gpu = gpuarray.to_gpu(a.T.copy())
    lda = m
    b = np.asarray(np.random.rand(m, n), np.float32)
    b_gpu = gpuarray.to_gpu(b.T.copy())
    ldb = m

    handle = cusolverDnCreate()
    Lwork = ctypes.c_int()

    cusolverDnSgetrf_bufferSize(handle, m, n, a_gpu, lda, Lwork)
    Workspace = gpuarray.empty(Lwork.value, dtype=np.float32)
    devIpiv = gpuarray.zeros(min(m, n), dtype=np.int32)
    devInfo = gpuarray.zeros(1, dtype=np.int32)

    cusolverDnSgetrf(handle, m, n, a_gpu, lda, Workspace, devIpiv, devInfo)
    if devInfo.get()[0] != 0:
        raise RuntimeError("error!")

    CUBLAS_OP_N = 0
    nrhs = n
    devInfo = gpuarray.zeros(1, dtype=np.int32)
    cusolverDnSgetrs(
        handle, CUBLAS_OP_N, n, nrhs, a_gpu, lda, devIpiv, b_gpu, ldb, devInfo
    )

    x_cusolver = b_gpu.get().T
    cusolverDnDestroy(handle)
    cusolve_time = time.time() - init_time
    print("cusolve time = %.6f" % cusolve_time)
    x_numpy = np.linalg.solve(a, b)
    numpy_time = time.time() - init_time - cusolve_time
    speedup = numpy_time / cusolve_time
    print("np.linalg.solve time = %.6f" % numpy_time)
    print("GPU speedup = %f" % speedup)
    print(np.allclose(x_numpy, x_cusolver, rtol=1e-02, atol=1e-04))
