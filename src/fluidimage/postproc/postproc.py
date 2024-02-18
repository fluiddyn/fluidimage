import os
import time
from warnings import warn

import numpy as np
import pylab

from fluidimage.data_objects.piv import LightPIVResults

from .util import compute_1dspectrum, compute_2dspectrum, compute_div, compute_rot

warn(
    "Functions and classes defined in this module (fluidimage.postproc.postproc)"
    "are deprecated (the code is not tested!). "
    "Please use fluidimage.postproc.vector_field instead.",
    DeprecationWarning,
)


def compute_ken(U, V):
    ken = (U**2 + V**2) / 2
    return ken


def compute_norm(U, V):
    norm = np.sqrt(U**2 + V**2)
    return norm


def compute_grid(xs, ys, deltaxs, deltays):
    x = np.unique(xs)
    y = np.unique(ys)
    X, Y = np.meshgrid(x, y)
    U = np.reshape(deltaxs, X.shape)
    V = np.reshape(deltays, X.shape)
    return X, Y, U, V


def compute_derivatives(dx, dy, U, V, edge_order=2):
    dUdx, dUdy = np.gradient(U, dx, dy, edge_order=edge_order)
    dVdx, dVdy = np.gradient(V, dx, dy, edge_order=edge_order)
    return dUdx, dUdy, dVdx, dVdy


def displayf(X, Y, U=None, V=None, background=None, *args):
    if background is not None:
        pylab.pcolor(X, Y, background)
    if U is not None:
        pylab.quiver(X, Y, U, V)


class DataObject:
    pass


class PIV_Postproc(LightPIVResults):
    def __init__(self, path):
        super().__init__(str_path=path)

        warn(
            "The class PIV_Postproc is deprecated (the code is not tested!). "
            "Please use fluidimage.postproc.vector_field.VectorFieldOnGrid instead.",
            DeprecationWarning,
        )

        self.path = os.path.abspath(path)

        self.X, self.Y, self.U, self.V = compute_grid(
            self.ixvecs_final,
            self.iyvecs_final,
            self.deltaxs_final,
            self.deltays_final,
        )

    def displayf(self, U=None, V=None, X=None, Y=None, bg=None, *args):
        if bg is None:
            bg = self.compute_norm(U, V)

        if X is None:
            X = self.X
            Y = self.Y

        if U is None:
            U = self.U
            V = self.V

        displayf(X, Y, U=U, V=V, background=bg, *args)

    def spatial_average(self, U):
        U_mean = np.mean(U, (0, 1))
        return U_mean

    def compute_derivatives(self, edge_order=2):
        dUdx, dUdy, dVdx, dVdy = compute_derivatives(
            self.dx, self.dy, self.U, self.V, edge_order=2
        )
        return dUdx, dUdy, dVdx, dVdy

    def compute_rot(self, edge_order=2):
        if not hasattr(self, "dUdx"):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(
                edge_order=edge_order
            )
        rot = compute_rot(self.dUdy, self.dVdx)
        return rot

    def compute_div(self, edge_order=2):
        if ~hasattr(self, "dUdx"):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(
                edge_order=edge_order
            )
        div = compute_div(self.dUdx, self.dVdy)
        return div

    def compute_ken(self):
        ken = compute_ken(self.U, self.V)
        return ken

    def compute_norm(self, U, V):
        norm = compute_norm(self.U, self.V)
        return norm

    def compute_spatial_fft(self, parseval=False):
        fftU, kx, ky, psdU = compute_2dspectrum(
            self.X, self.Y, self.U, axes=(0, 1)
        )
        fftV, kx, ky, psdV = compute_2dspectrum(
            self.X, self.Y, self.V, axes=(0, 1)
        )

        if not hasattr(self, "fft"):
            self.fft = DataObject()
        if not hasattr(self.fft, "spatial"):
            spatial = DataObject()
            self.fft.spatial = spatial

        self.fft.spatial.kx = kx
        self.fft.spatial.ky = ky
        self.fft.spatial.fftU = fftU
        self.fft.spatial.fftV = fftV
        self.fft.spatial.psdU = psdU
        self.fft.spatial.psdV = psdV
        # if parseval:
        #     dx = self.X[1][0] - self.X[0][0]
        #     dy = self.Y[0][1] - self.Y[0][0]
        #     Lx = np.max(self.X) - np.min(self.X)
        #     Ly = np.max(self.Y) - np.min(self.Y)
        #     dkx = kx[1] - kx[0]
        #     dky = ky[1] - ky[0]
        #     energphys = np.sum(self.U ** 2 + self.V ** 2) * dx * dy / Lx / Ly
        #     energspectral = (
        #         np.sum(self.fft.spatial.psdU + self.fft.spatial.psdV) * dkx * dky
        #     )
        #     print("%%%% PARSEVAL %%%%")
        #     print("np.sum(U**2+V**2) * dx*dy / Lx/Ly =")
        #     print(energphys)
        #     print("np.sum(psd) * dkx*dky=")
        #     print(energspectral)


class PIV_PostProc_serie(LightPIVResults):
    def __init__(self, path=None):
        warn(
            "The class PIV_Postproc_serie is deprecated (the code is not tested!). "
            "Please use fluidimage.postproc.vector_field.ArrayOfVectorFieldOnGrid instead.",
            DeprecationWarning,
        )

        self.path = path
        path0 = path[0]
        super().__init__(str_path=path0)
        for ti, pathi in enumerate(path[1:]):
            tmp = PIV_Postproc(path=pathi)
            self.deltaxs_final = np.vstack(
                [self.deltaxs_final, tmp.deltaxs_final]
            )
            self.deltays_final = np.vstack(
                [self.deltays_final, tmp.deltays_final]
            )
        self.X, self.Y, self.dx, self.dy, self.U, self.V = self.compute_grid()

    def set_time(self, dt):
        nb_files = len(self.path)
        self.t = np.linspace(0, dt * (nb_files - 1), nb_files)

    def displayf(
        self, U=None, V=None, bg=None, X=None, Y=None, timesleep=0.5, *args
    ):
        if U is None and bg.ndim == 3:
            U = V = [None] * len(bg)
        elif bg is None and U.ndim == 3:
            bg = self.compute_norm(U, V)

        if X is None:
            X = self.X
            Y = self.Y

        if U is None:
            displayf(X, Y, U=U, V=V, background=bg, *args)
        else:
            for i in range(len(U)):
                displayf(X, Y, U=U[i], V=V[i], background=bg[i], *args)
                pylab.show()
                time.sleep(timesleep)

    def time_average(self, U):
        U_mean = np.mean(U, 0)
        return U_mean

    def spatial_average(self, U):
        U_mean = np.mean(U, (1, 2))
        return U_mean

    def compute_grid(self):
        U = [None] * len(self.path)
        V = [None] * len(self.path)
        for ti, pathi in enumerate(self.path):
            X, Y, dx, dy, U[ti], V[ti] = compute_grid(
                self.ixvecs_final,
                self.iyvecs_final,
                self.deltaxs_final[ti],
                self.deltays_final[ti],
            )
        return X, Y, dx, dy, U, V

    def compute_derivatives(self, edge_order=2):
        dUdx = np.zeros(np.shape(self.U))
        dUdy = np.zeros(np.shape(self.U))
        dVdx = np.zeros(np.shape(self.U))
        dVdy = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate(self.path):
            dUdx[ti], dUdy[ti], dVdx[ti], dVdy[ti] = compute_derivatives(
                self.dx, self.dy, self.U[ti], self.V[ti], edge_order=2
            )
        return dUdx, dUdy, dVdx, dVdy

    def compute_rot(self, edge_order=2):
        if not hasattr(self, "dUdx"):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(
                edge_order=edge_order
            )
        rot = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate(self.path):
            rot[ti] = compute_rot(self.dUdy[ti], self.dVdx[ti])
        return rot

    def compute_div(self, edge_order=2):
        if ~hasattr(self, "dUdx"):
            self.dUdx, self.dUdy, self.dVdx, self.dVdy = self.compute_derivatives(
                edge_order=edge_order
            )
        div = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate(self.path):
            div[ti] = compute_div(self.dUdx[ti], self.dVdy[ti])
        return div

    def compute_ken(self):
        ken = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate(self.path):
            ken[ti] = compute_ken(self.U[ti], self.V[ti])
        return ken

    def compute_norm(self, U, V):
        norm = np.zeros(np.shape(self.U))
        for ti, pathi in enumerate(self.path):
            norm[ti] = compute_norm(U[ti], V[ti])
        return norm

    def compute_temporal_fft(self):
        if not hasattr(self, "t"):
            print("please define time before perform temporal Fourier transform")
        else:
            fftU, omega, psdU = compute_1dspectrum(self.t, self.U, axis=0)
            fftV, omega, psdV = compute_1dspectrum(self.t, self.V, axis=0)

            if not hasattr(self, "fft"):
                self.fft = DataObject()
            if not hasattr(self.fft, "time"):
                time = DataObject()
                self.fft.time = time

            self.fft.time.omega = omega
            self.fft.time.fftU = fftU
            self.fft.time.fftV = fftV
            self.fft.time.psdU = psdU
            self.fft.time.psdV = psdV
            # if parseval:
            #     # parseval
            #     dt = self.t[1] - self.t[0]
            #     Lt = np.max(self.t) - np.min(self.t)
            #     domega = omega[1] - omega[0]
            #     energphys = np.sum(self.U ** 2 + self.V ** 2) * dt / Lt
            #     energspectral = (
            #         np.sum(self.fft.time.psdU + self.fft.time.psdV) * domega
            #     )
            #     print("%%%% PARSEVAL %%%%")
            #     print("np.sum(U**2+V**2) * dt / Lt =")
            #     print(energphys)
            #     print("np.sum(psd) * domega=")
            #     print(energspectral)

    def compute_spatial_fft(self):
        fftU, kx, ky, psdU = compute_2dspectrum(
            self.X, self.Y, self.U, axes=(1, 2)
        )
        fftV, kx, ky, psdV = compute_2dspectrum(
            self.X, self.Y, self.V, axes=(1, 2)
        )

        if not hasattr(self, "fft"):
            self.fft = DataObject()
        if not hasattr(self.fft, "spatial"):
            spatial = DataObject()
            self.fft.spatial = spatial

        self.fft.spatial.kx = kx
        self.fft.spatial.ky = ky
        self.fft.spatial.fftU = fftU
        self.fft.spatial.fftV = fftV
        self.fft.spatial.psdU = psdU
        self.fft.spatial.psdV = psdV
        # if parseval:
        #     dx = self.X[1][0] - self.X[0][0]
        #     dy = self.Y[0][1] - self.Y[0][0]
        #     Lx = np.max(self.X) - np.min(self.X)
        #     Ly = np.max(self.Y) - np.min(self.Y)
        #     dkx = kx[1] - kx[0]
        #     dky = ky[1] - ky[0]
        #     energphys = np.sum(self.U ** 2 + self.V ** 2) * dx * dy / Lx / Ly
        #     energspectral = (
        #         np.sum(self.fft.spatial.psdU + self.fft.spatial.psdV) * dkx * dky
        #     )
        #     print("%%%% PARSEVAL %%%%")
        #     print("np.sum(U**2+V**2) * dx*dy / Lx/Ly =")
        #     print(energphys)
        #     print("np.sum(psd) * dkx*dky=")
        #     print(energspectral)

    def compute_spatiotemp_fft(self):
        if hasattr(self, "fft.spatial"):
            fftU, omega, psdU = compute_1dspectrum(
                self.t, self.fft.spatial.fftU, axis=0
            )
            fftV, omega, psdV = compute_1dspectrum(
                self.t, self.fft.spatial.fftV, axis=0
            )
            kx = self.fft.spatial.kx
            ky = self.fft.spatial.ky
            Lkx = np.max(kx) - np.min(kx)
            nx = np.shape(self.X)[0]
            Lky = np.max(ky) - np.min(ky)
            ny = np.shape(self.Y)[0]
            psdU *= 1.0 / Lkx / nx / Lky / ny
            psdV *= 1.0 / Lkx / nx / Lky / ny
        elif hasattr(self, "fft.time"):
            fftU, kx, ky, psdU = compute_2dspectrum(
                self.X, self.Y, self.fft.time.fftU, axes=(1, 2)
            )
            fftV, kx, ky, psdV = compute_2dspectrum(
                self.X, self.Y, self.fft.time.fftV, axes=(1, 2)
            )
            omega = self.fft.time.omega
            Lomega = np.max(omega) - np.min(omega)
            n = np.shape(self.t)[0]
            psdU *= 1.0 / Lomega / n
            psdV *= 1.0 / Lomega / n
        else:
            self.compute_temporal_fft()
            fftU, kx, ky, psdU = compute_2dspectrum(
                self.X, self.Y, self.fft.time.fftU, axes=(1, 2)
            )
            fftV, kx, ky, psdV = compute_2dspectrum(
                self.X, self.Y, self.fft.time.fftV, axes=(1, 2)
            )
            omega = self.fft.time.omega
            Lomega = np.max(omega) - np.min(omega)
            n = np.shape(self.t)[0]
            psdU *= 1.0 / Lomega / n
            psdV *= 1.0 / Lomega / n

        if not hasattr(self, "fft"):
            self.fft = DataObject()
        if not hasattr(self.fft, "spatiotemp"):
            spatiotemp = DataObject()
            self.fft.spatiotemp = spatiotemp

        self.fft.spatiotemp.omega = omega
        self.fft.spatiotemp.kx = kx
        self.fft.spatiotemp.ky = ky
        self.fft.spatiotemp.fftU = fftU
        self.fft.spatiotemp.fftV = fftV
        self.fft.spatiotemp.psdU = psdU
        self.fft.spatiotemp.psdV = psdV

        # if parseval:
        #     dx = self.X[1][0] - self.X[0][0]
        #     dy = self.Y[0][1] - self.Y[0][0]
        #     Lx = np.max(self.X) - np.min(self.X)
        #     Ly = np.max(self.Y) - np.min(self.Y)
        #     dkx = kx[1] - kx[0]
        #     dky = ky[1] - ky[0]
        #     dt = self.t[1] - self.t[0]
        #     Lt = np.max(self.t) - np.min(self.t)
        #     domega = omega[1] - omega[0]
        #     energphys = (
        #         np.sum(self.U ** 2 + self.V ** 2) * dx * dy * dt / Lx / Ly / Lt
        #     )
        #     energspectral = (
        #         np.sum(self.fft.spatiotemp.psdU + self.fft.spatiotemp.psdV)
        #         * dkx
        #         * dky
        #         * domega
        #     )
        #     print("%%%% PARSEVAL %%%%")
        #     print("np.sum(U**2+V**2) * dx*dy*dt / Lx/Ly/Lt =")
        #     print(energphys)
        #     print("np.sum(psd) * dkx*dky*domega=")
        #     print(energspectral)
