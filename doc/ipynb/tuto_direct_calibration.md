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

```{code-cell} ipython3
%matplotlib inline
import numpy as np
```

# Direct Calibration

For this tutorial we will use files contained in the fluidimage repository. We assume that fluidimage has been installed with `python setup.py develop` and we deduce the path of the repository from the path of the package fluidimage:

```{code-cell} ipython3
from fluidimage import get_path_image_samples
```

We first import the class `CalibDirect`

```{code-cell} ipython3
from fluidimage.calibration import CalibDirect
```

Give number of pixels of images and the path of grid points made by UVMAT

```{code-cell} ipython3
path_cam = (get_path_image_samples() /
            '4th_PIV-Challenge_Case_E/E_Calibration_Images/Camera_01')
glob_str_xml = path_cam / 'img*.xml'
shape_img = 1024, 1024
calib = CalibDirect(glob_str_xml, shape_img)
```

We now compute interpolents able to compute the physical coordinates from the indices in the images and  (for each level)

```{code-cell} ipython3
calib.compute_interpolents()
```

The quality of this step can be checked with the function `check_interp_levels`

```{code-cell} ipython3
calib.check_interp_levels()
```

And finally, we compute the interpolents for equations of optical paths from nbline_x * nbline_y lines and save the calibration

```{code-cell} ipython3
nb_lines_x, nb_lines_y = 128, 64
calib.compute_interpolents_pixel2line(nb_lines_x, nb_lines_y, test=False)
```

```{code-cell} ipython3
calib.save(path_cam / 'calib1.npy')
```

It is very important to check that everything seems all right because it is easy to make something wrong with such calibration.

```{code-cell} ipython3
calib.check_interp_lines()
```

```{code-cell} ipython3
calib.check_interp_lines_coeffs()
```

```{code-cell} ipython3
from fluidimage.calibration.util import get_plane_equation
z0 = 0.05
alpha = 0
beta = 0
a, b, c, d = get_plane_equation(z0, alpha, beta)
```

```{code-cell} ipython3
indx = np.arange(100, 1000, 100)
indy = np.arange(100, 1000, 100)

calib.intersect_with_plane(indx, indy, a, b, c, d)
```

```{code-cell} ipython3
dx = np.random.randint(0, 5, indx.size)
dy = np.random.randint(0, 5, indx.size)
calib.apply_calib(indx, indy, dx, dy, a, b, c, d)
```
