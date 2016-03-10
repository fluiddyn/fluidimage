import h5py
from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv2 import WorkPIV
from fluidimage.data_objects.piv import (LightPIVResults)

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
import pylab

postp=PIV_Postproc(path='piv_Oseen_center01-02_light.h5')

rot=postp.compute_rot()

div=postp.compute_div()

pylab.figure;
postp.displayf(U=postp.U, V=postp.V, bg=div) 

pylab.figure;
postp.displayf(bg=rot) 

#%%
from postproc import PIV_PostProc_serie
import pylab

postp=PIV_PostProc_serie(path=['piv_Oseen_center01-02_light.h5','piv_Oseen_center01-02_light.h5'])

rot=postp.compute_rot()

div=postp.compute_div()

pylab.figure;
postp.displayf(U=postp.U, V=postp.V, bg=div) 

pylab.figure;
postp.displayf(bg=rot) 

pylab.figure;
postp.displayf(U=postp.U, V=postp.V) 