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

# Postprocess PIV results (simple calibration)

```{code-cell} ipython3
import h5py
import numpy as np
```

```{code-cell} ipython3
from fluidimage import get_path_image_samples
```

```{code-cell} ipython3
from fluidimage.calibration import Calibration2DSimple
```

```{code-cell} ipython3
path_base = get_path_image_samples() / 'Milestone'
path_piv = path_base / 'piv_0000a-b.h5'
```

## Create the calibration object

```{code-cell} ipython3
with h5py.File(path_piv) as file:
    shape_im = file['couple/shape_images'][...]

point0 = (10, 20)
point1 = (10, 10)

calib = Calibration2DSimple(point0, point1, distance=0.01, aspect_ratio_pixel=1.0,
                            shape_image=shape_im, point_origin=[i//2 for i in shape_im])
```

```{code-cell} ipython3
print(calib.yphys1pixel)
```

## Application to synthetic data

We can try the calibration object on synthetic data:

```{code-cell} ipython3
ixs = np.linspace(0, 100, 11)
iys = np.linspace(0, 100, 11)
xphys, yphys = calib.pix2phys(ixs, iys)
print(xphys, yphys, sep='\n')
```

```{code-cell} ipython3
dxs = np.random.randint(-5, 5, ixs.size)
dys = np.random.randint(-5, 5, ixs.size)
xphys, yphys, dxphys, dyphys = calib.displ2phys(ixs, iys, dxs, dys)
print(xphys, yphys, dxphys, dyphys, sep='\n')
```

## Application to real data

```{code-cell} ipython3
with h5py.File(path_piv) as file:
    deltaxs_final = file['piv1/deltaxs_final'][:]
    deltays_final = file['piv1/deltays_final'][:]
    ixvecs_final = file['piv1/ixvecs_final'][:]
    iyvecs_final = file['piv1/iyvecs_final'][:]
```

```{code-cell} ipython3
xphys, yphys, dxphys, dyphys = calib.displ2phys(
    ixvecs_final, iyvecs_final, deltaxs_final, deltays_final)
```

```{code-cell} ipython3
print(xphys, yphys, sep='\n')
```

## 2d grid

+++

We need to produce a good grid in the physical space to interpolate the data:

```{code-cell} ipython3
from fluidimage.postproc.piv import get_grid_pixel_from_piv_file

xs1d, ys1d = get_grid_pixel_from_piv_file(path_piv)

print(xs1d, ys1d, sep='\n')
```

```{code-cell} ipython3
print(len(xs1d), len(ys1d))
```

```{code-cell} ipython3
xs2d, ys2d = np.meshgrid(xs1d, ys1d)

assert xs2d.shape == ys2d.shape
print(xs2d.shape)
```

```{code-cell} ipython3
print(xs2d)
```

```{code-cell} ipython3
print(ys2d)
```

```{code-cell} ipython3
xs2dphys, ys2dphys = calib.pix2phys(xs2d, ys2d)
```

```{code-cell} ipython3
print(xs2dphys)
```

```{code-cell} ipython3
print(ys2dphys)
```

We define a function to interpolate the data on the grid

```{code-cell} ipython3
from scipy.interpolate import griddata

def gridd(delta):
        delta_grid = griddata((yphys, xphys), delta, (ys2dphys, xs2dphys), method='cubic')
        return delta_grid.astype('float32')
```

```{code-cell} ipython3
dxphys_grid = gridd(dxphys)
dyphys_grid = gridd(dyphys)
```

```{code-cell} ipython3
print(dxphys_grid)
```

We can finally produce PIVOnGrid objects...
