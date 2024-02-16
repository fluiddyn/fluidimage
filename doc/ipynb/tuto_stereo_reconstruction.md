---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Stereo Reconstruction with Direct Calibration

We present in this notebook how to do stereo reconstruction with 2 cameras.

```{code-cell} ipython3
# to handle the paths
import os
from fluidimage import get_path_image_samples

# to load the PIV files
import h5py

# the tools for stereo reconstruction:
from fluidimage.calibration import DirectStereoReconstruction
from fluidimage.calibration.util import get_plane_equation

pathbase = get_path_image_samples() / '4th_PIV-Challenge_Case_E'
```

```{code-cell} ipython3
def make_calib_path(index_cam):
    path_cam = pathbase / f'E_Calibration_Images/Camera_0{index_cam}'
    path = path_cam / f'calib{index_cam}.npz'
    if not os.path.exists(path):
        # we need to create this calib file.
        from fluidimage.calibration import CalibDirect
        calib = CalibDirect(
            glob_str_xml=str(path_cam / 'img*.xml'),
            shape_img=(1024, 1024))
        calib.compute_interpolents()
        nb_lines_x, nb_lines_y = 64, 64
        calib.compute_interpolents_pixel2line(nb_lines_x, nb_lines_y, test=False)
        calib.save(path)
    return path
```

Make an instance of the class with the 2 calibration files

```{code-cell} ipython3
paths_calibs = [make_calib_path(index_cam) for index_cam in [1, 3]]
stereo = DirectStereoReconstruction(*paths_calibs)
```

Define the plane of measurement

```{code-cell} ipython3
z0 = 0
alpha = 0
beta = 0
a, b, c, d = get_plane_equation(z0, alpha, beta)
```

Get the 2 PIV fields

```{code-cell} ipython3
def get_piv_field(path):

    try:
        with h5py.File(path, 'r') as file:
            keyspiv = [key for key in file.keys() if key.startswith('piv')]
            keyspiv.sort()
            key = keyspiv[-1]
            X = file[key]['xs'][:]
            Y = file[key]['ys'][:]
            dx = file[key]['deltaxs_final'][:]
            dy = file[key]['deltays_final'][:]
    except Exception:
        print(path)
        raise

    return X, Y, dx, dy

postfix = '.piv'
v = 'piv_00001-00002.h5'

path1 = pathbase / ('E_Particle_Images/Camera_01' + postfix) / v
path3 = pathbase / ('E_Particle_Images/Camera_03' + postfix) / v

Xl, Yl, dxl, dyl = get_piv_field(path1)
Xr, Yr, dxr, dyr = get_piv_field(path3)
```

Apply calibration on the 2 cameras, the result is given of their respectives planes

```{code-cell} ipython3
X0, X1, d0cam, d1cam = stereo.project2cam(
    Xl, Yl, dxl, dyl, Xr, Yr, dxr, dyr, a, b, c, d, check=False)
```

Find the common grid

```{code-cell} ipython3
X, Y, Z = stereo.find_common_grid(X0, X1, a, b, c, d)
```

Reconstruct the 3 components of the velocity

```{code-cell} ipython3
dx, dy, dz, erx, ery, erz = stereo.reconstruction(
    X0, X1, d0cam, d1cam, a, b, c, d, X, Y, check=False)
```
