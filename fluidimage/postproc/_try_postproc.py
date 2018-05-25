
import h5py
from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv import WorkPIV
import numpy as np
import pylab

# %%
params = WorkPIV.create_default_params()

# for a very short computation
params.piv0.shape_crop_im0 = 24
params.piv0.grid.overlap = 0.

params.piv0.method_subpix = "2d_gaussian"
params.piv0.method_correl = "fftw"

params.multipass.number = 3
# params.multipass.use_tps = 'last'

piv = WorkPIV(params=params)

series = SeriesOfArrays("../../image_samples/Karman/Images", "i+1:i+3")
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)
result.save()
# result.display()

lightresult = result.make_light_result()

lightresult.save()


# %%
from postproc import PIV_Postproc

postp = PIV_Postproc(path="piv_Karman01-02_light.h5")

rot = postp.compute_rot()

div = postp.compute_div()

pylab.figure
postp.displayf(X=postp.X, Y=postp.Y, U=postp.U, V=postp.V)

pylab.figure
postp.displayf(bg=rot)

# %%
from postproc import PIV_PostProc_serie

postp = PIV_PostProc_serie(path=["piv_Karman01-02_light.h5"] * 10)
postp.set_time(np.linspace(0, 10, 10))

rot = postp.compute_rot()

div = postp.compute_div()

postp.compute_temporal_fft()

pylab.figure
postp.displayf(U=postp.U, V=postp.V, bg=div)

pylab.figure
postp.displayf(bg=rot)

pylab.figure
postp.displayf(U=postp.U, V=postp.V)


# tests Fourier transform
from postproc import PIV_PostProc_serie

Nt = 100
postp = PIV_PostProc_serie(path=["piv_Karman01-02_light.h5"] * Nt)

t = np.linspace(0, 1000, Nt)

postp.set_time(t)

kx = 0.05
ky = 0.17
omega = 0.02

for i in range(np.shape(postp.U)[0]):
    postp.U[i] = np.cos(omega * t[i] + kx * postp.X + ky * postp.Y)
    postp.V[i] = np.cos(omega * t[i] + kx * postp.X + ky * postp.Y)


# FFT temporelle
postp.compute_temporal_fft(parseval=True)
omega = postp.fft.time.omega
psd = postp.fft.time.psdU + postp.fft.time.psdV
pylab.loglog(omega, postp.spatial_average(psd))

# FFT spatiale
postp.compute_spatial_fft(parseval=True)
Kx, Ky = np.meshgrid(postp.fft.spatial.kx, postp.fft.spatial.ky)
Kx = Kx.transpose()
Ky = Ky.transpose()
psd = postp.fft.spatial.psdU + postp.fft.spatial.psdV
postp.displayf(X=Kx, Y=Ky, bg=np.log(postp.time_average(psd)))


# FFT spatiotemporelle
postp.compute_spatiotemp_fft(parseval=True)
Kx, Ky = np.meshgrid(postp.fft.spatiotemp.kx, postp.fft.spatiotemp.ky)
Kx = Kx.transpose()
Ky = Ky.transpose()
omega = postp.fft.spatiotemp.omega
psd = postp.fft.spatiotemp.psdU + postp.fft.spatiotemp.psdV
psd = psd[0:10:]
postp.displayf(X=Kx, Y=Ky, bg=np.log(psd))


# useful quantities
X = postp.X
Y = postp.Y
dx = X[1][0] - X[0][0]
dy = Y[0][1] - Y[0][0]
Lx = np.max(X) - np.min(X)
Ly = np.max(Y) - np.min(Y)
nx = X.shape[0]
ny = X.shape[1]

kx = postp.fft.spatiotemp.kx
ky = postp.fft.spatiotemp.ky
dkx = kx[1] - kx[0]
dky = ky[1] - ky[0]
Kx, Ky = np.meshgrid(postp.fft.spatiotemp.kx, postp.fft.spatiotemp.ky)
Kx = Kx.transpose()
Ky = Ky.transpose()

omega = postp.fft.spatiotemp.omega
domega = omega[1] - omega[0]
dt = postp.t[1] - postp.t[0]
Lt = np.max(postp.t) - np.min(postp.t)


# moyenne spatial de fft spatiotemp vs fft temp

psd = (
    np.sum(postp.fft.spatiotemp.psdU + postp.fft.spatiotemp.psdV, (1, 2))
    * 1.0
    * dkx
    * dky
)
pylab.loglog(omega, psd)
psd2 = postp.fft.time.psdU + postp.fft.time.psdV
psd2 = np.sum(psd2, (1, 2)) * 1.0 * (dx * dy) / (Lx * Ly)
pylab.loglog(omega, psd2, "r+")
print(np.mean(((psd2 - psd) ** 2) / psd))
print(np.max(((psd2 - psd) ** 2) / psd))

# moyenne temporelle de fft spatiotemp vs fft spatial
psd = (
    np.sum(postp.fft.spatiotemp.psdU + postp.fft.spatiotemp.psdV, 0)
    * 1.0
    * domega
)
pylab.pcolor(Kx, Ky, psd)
psd2 = postp.fft.spatial.psdU + postp.fft.spatial.psdV
psd2 = np.sum(psd2, 0) * 1.0 * dt / Lt
pylab.pcolor(Kx, Ky, psd2)
pylab.pcolor(Kx, Ky, psd2 - psd)
print(np.mean(((psd2 - psd) ** 2) / psd))
print(np.max(((psd2 - psd) ** 2) / psd))
