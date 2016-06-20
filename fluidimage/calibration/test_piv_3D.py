# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:32:26 2016

@author: campagne8a
"""


from fluidimage.topologies.piv import TopologyPIV
from fluidimage.data_objects.piv import HeavyPIVResults
# from fluidimage.calibration_3d.
from get_calib import calib_parameters_from_uvmat
# from fluidimage.calibration_3d.
from apply_calib import pix2phys, pix2phys_UV

params = TopologyPIV.create_default_params()
params.piv0.shape_crop_im0 = 64

params.fix.displacement_max = None
params.fix.correl_min = 0.1

params.piv0.grid.overlap=50
params.piv0.grid.overlap = 0

params.saving.how= 'recompute'

# path = '../../image_samples/Oseen/Images/Oseen_center*'
# path = '../../image_samples/Karman/Images'

for i in range(10):
    path = (
        '/.fsdyn_people/campagne8a/project/'
        '15DELDUCA/tmp/Antoine/piv_3D/test2/'
        'data_reorganized/level{}').format(i+1)
    print(path)
# path = '../../image_samples/Jet/Images/c*'
# params.series.strcouple = 'i+60, 0:2'
# params.series.strcouple = 'i+60:i+62, 0'

    params.series.path = path

    topology = TopologyPIV(params)

    topology.compute(sequential=False)

# topology.make_code_graphviz('topo.dot')
# then the graph can be produced with the command:
# dot topo.dot -Tpng -o topo.png
# dot topo.dot -Tx11

#% movie at 1 level
p =[]
for i in range(18):
    p.append(HeavyPIVResults(str_path='/.fsdyn_people/campagne8a/project/15DELDUCA/tmp/Antoine/piv_3D/test2/data_reorganized/level1.piv/piv_im{}-{}.h5'.format(i+1,i+2)))

for pi in p:
    pi.display()
#%% quiver 3D at a given time
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np


calib = calib_parameters_from_uvmat('/.fsdyn_people/campagne8a/project/15DELDUCA/tmp/Antoine/piv_3D/Calibration/Calib.xml')

p=[]
for i in range(0,9,2):
    pi = HeavyPIVResults(str_path='/.fsdyn_people/campagne8a/project/15DELDUCA/tmp/Antoine/piv_3D/test2/data_reorganized/level{}.piv/piv_im2-3.h5'.format(i+1))
    pi.level = i + 1    
    p.append(pi)
    
X=[];
Y=[]
VX=[]
VY=[]    
Z = []
VZ=[]
for pi in p:
    y = pi.ys.max() - pi.ys  
    x = pi.xs.max() - pi.xs  

    Xp, Yp, Zp, dxp, dyp, dzp = pix2phys_UV(calib, pi.xs , pi.ys, pi.deltaxs, pi.deltays, pi.level)

    X=np.hstack([X, Xp])
    Y=np.hstack([Y, Yp])
    Z=np.hstack([Z, Zp])
    VX=np.hstack([VX, dxp])
    VY=np.hstack([VY, dyp])
    VZ=np.hstack([VZ, dzp])
    
fig = plt.figure()
ax = fig.gca(projection='3d')

ind = range(0,Z.size, 3)
ax.quiver3D(Z[ind], X[ind], Y[ind],VZ[ind], VX[ind], VY[ind], length=0.02)
#pylab.plot(pi.xs,Zp,'+')
ax.set_xlabel('z (m)')
ax.set_ylabel('x (m)')
ax.set_zlabel('y (m)')

