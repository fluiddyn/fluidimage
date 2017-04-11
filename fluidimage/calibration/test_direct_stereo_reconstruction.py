import h5py
from fluidimage.calibration.util import get_plane_equation
from fluidimage.calibration import DirectStereoReconstruction
import os
import fluidimage

def get_piv_field(path):

    try:
        with h5py.File(path, 'r') as f:
            keyspiv = [key for key in f.keys() if key.startswith('piv')]
            keyspiv.sort()
            key = keyspiv[-1]
            X = f[key]['xs'].value
            Y = f[key]['ys'].value
            dx = f[key]['deltaxs_final'].value
            dy = f[key]['deltays_final'].value
    except Exception:
        print(path)
        raise

    return X, Y, dx, dy

path_fluidimage = os.path.join(
    '/', *os.path.abspath(fluidimage.__file__).split('/')[:-2])

postfix = '.piv/'

pathbase = path_fluidimage + '/image_samples/4th_PIV-Challenge_Case_E'

# level = 0
v = 'piv_00001-00002.h5'

pathcalib1 = pathbase + '/E_Calibration_Images/Camera_01/calib1.npy'
pathcalib3 = pathbase + '/E_Calibration_Images/Camera_03/calib3.npy'

dt = 0.001

path1 = pathbase + '/E_Particle_Images/Camera_01' + postfix + v
path3 = pathbase + '/E_Particle_Images/Camera_03' + postfix + v

z0 = 0
alpha = 0
beta = 0
a, b, c, d = get_plane_equation(z0, alpha, beta)

Xl, Yl, dxl, dyl = get_piv_field(path1)
Xr, Yr, dxr, dyr = get_piv_field(path3)

stereo = DirectStereoReconstruction(pathcalib1, pathcalib3)
X0, X1, d0cam, d1cam = stereo.project2cam(
    Xl, Yl, dxl, dyl, Xr, Yr, dxr, dyr, a, b, c, d, check=False)
X, Y, Z = stereo.find_common_grid(X0, X1, a, b, c, d)

dx, dy, dz, erx, ery, erz = stereo.reconstruction(
    X0, X1, d0cam, d1cam, a, b, c, d, X, Y, check=False)
dx, dy, dz, erx, ery, erz = dx/dt, dy/dt, dz/dt, erx/dt, ery/dt, erz/dt
