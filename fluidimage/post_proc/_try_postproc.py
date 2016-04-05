import h5py
from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv2 import WorkPIV
from fluidimage.data_objects.piv import (LightPIVResults)
import numpy as np
import pylab

#%%
params = WorkPIV.create_default_params()

# for a very short computation
params.piv0.shape_crop_im0 = 24
params.piv0.grid.overlap = 0.

params.piv0.method_subpix = 'centroid'
params.piv0.method_correl = 'fftw'

params.multipass.number = 2
params.multipass.use_tps = True

piv = WorkPIV(params=params)

series = SeriesOfArrays('../../image_samples/Oseen/Images', 'i+1:i+3')
serie = series.get_serie_from_index(0)

result = piv.calcul(serie)

result.display()

lightresult=LightPIVResults(result.piv1.deltaxs_approx,
                            result.piv1.deltays_approx,result.piv1.ixvecs_grid,
                            result.piv1.iyvecs_grid,couple=result.piv1.couple,
                            params=result.piv1.params)
lightresult.save()


#%%
from postproc import PIV_Postproc

postp=PIV_Postproc(path='piv_Oseen_center01-02_light.h5')

rot=postp.compute_rot()

div=postp.compute_div()

pylab.figure;
postp.displayf(U=postp.U, V=postp.V, bg=div) 

pylab.figure;
postp.displayf(bg=rot) 

#%%
from postproc import PIV_PostProc_serie

postp=PIV_PostProc_serie(path=['piv_Oseen_center01-02_light.h5']*10)
postp.set_time(np.linspace(0,10,10))

rot=postp.compute_rot()

div=postp.compute_div()

postp.compute_temporal_fft()

pylab.figure;
postp.displayf(U=postp.U, V=postp.V, bg=div) 

pylab.figure;
postp.displayf(bg=rot) 

pylab.figure;
postp.displayf(U=postp.U, V=postp.V) 



#%%
from postproc import PIV_PostProc_serie

postp=PIV_PostProc_serie(path=['piv_Oseen_center01-02_light.h5']*1000)
postp.set_time(np.linspace(0,10,10))

t=np.linspace(0,1000,1000)
kx = 0.05
ky = 0.17
omega = 0.02

for i in range(np.shape(postp.U)[0]):
    postp.U[i] = np.cos(omega*t[i]+kx*postp.X+ky*postp.Y)
    postp.U[i] = np.cos(omega*t[i]+kx*postp.X+ky*postp.Y)


#% FFT temporelle
postp.compute_temporal_fft()
omega = postp.fft.time.omega
psd = postp.fft.time.psdU+postp.fft.time.psdV
pylab.loglog(omega, postp.spatial_average(psd))

#% FFT spatiale
postp.compute_spatial_fft()
Kx, Ky = np.meshgrid(postp.fft.spatial.kx, postp.fft.spatial.ky)
Kx = Kx.transpose()
Ky = Ky.transpose()
psd = postp.fft.spatial.psdU+postp.fft.spatial.psdV
postp.displayf(X=Kx, Y=Ky, bg = np.log(postp.time_average(psd)))

#% FFT spatiotemporelle
postp.compute_spatiotemp_fft()
Kx, Ky = np.meshgrid(postp.fft.spatiotemp.kx, postp.fft.spatiotemp.ky)
Kx = Kx.transpose()
Ky = Ky.transpose()
omega = postp.fft.spatiotemp.omega
psd = postp.fft.spatiotemp.psdU+postp.fft.spatiotemp.psdV
psd=psd[0:10:]
postp.displayf(X=Kx, Y=Ky, bg = np.log(psd))

# moyenne temporelle de fft spatiotemp vs fft temp
X=postp.X
Y=postp.Y
nx = X.shape[0]
ny = X.shape[1]    
dx = X[1][0]-X[0][0]
dy = Y[0][1]-Y[0][0]
Lx = np.max(X) - np.min(X)
Ly = np.max(Y) - np.min(Y)
kx=postp.fft.spatiotemp.kx
ky=postp.fft.spatiotemp.ky
dkx=kx[1]-kx[0]
dky=ky[1]-ky[0]
nkx=kx.size
nky=ky.size
Lkx=np.max(kx) - np.min(kx)
Lky=np.max(ky) - np.min(ky)
omega = postp.fft.spatiotemp.omega

psd=np.sum(postp.fft.spatiotemp.psdU+postp.fft.spatiotemp.psdV,(1,2))*dkx*dky/Lkx/Lky
max1=psd.max()
pylab.loglog(omega, psd)
psd2 = postp.fft.time.psdU+postp.fft.time.psdV
psd2 = np.sum(psd2, (1,2))*(1.0*nx*ny*dx*dy/Lx/Ly)
max2=psd2.max()
pylab.loglog(omega, psd2, 'r+')
