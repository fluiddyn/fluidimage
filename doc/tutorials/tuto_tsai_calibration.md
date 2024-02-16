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

# Tsai calibration
Get first the calibration given by UVMAT

```{code-cell} ipython3
from fluidimage import get_path_image_samples
```

```{code-cell} ipython3
import h5py
from fluidimage.calibration import Calibration

pathbase = get_path_image_samples() / 'Milestone'
path_calib = pathbase / 'PCO_top.xml'
calib = Calibration(path_calib)
```

Get the velocity field

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

path_v = pathbase / 'piv_0000a-b.h5'

X, Y, dx, dy = get_piv_field(path_v)
```

Apply calibration, number of pixels in the y direction has to be given

```{code-cell} ipython3
nbypix = 2160

Xphys, Yphys, Zphys, dxphys, dyphys, dzphys = calib.pix2phys_UV(
    X, Y, dx, dy, index_level=0, nbypix=nbypix)
```
