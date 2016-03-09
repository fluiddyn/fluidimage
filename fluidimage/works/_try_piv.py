
import h5netcdf
import h5py
from fluiddyn.util.serieofarrays import SeriesOfArrays
from fluidimage.works.piv2 import WorkPIV

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

result.save()

#f=h5netcdf.File('piv_Oseen_center01-02.h5')
f=h5py.File('piv_Oseen_center01-02.h5')
