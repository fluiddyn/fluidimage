---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 3D Calibration using OpenCV

The calibration was performed using the Python bindings of the OpenCV [Bradski, 2000] library. This has been
made easier to use through the module `fluidimage.calibration.calib_cv`. We shall use this module on a set of
5 calibration images of a target which has a circle grid. The white dots on this particular target is evenly
spaced at `3 mm`. For the same camera position, the target coordinate `z` varies as `[-6, -3, 0, 3, 6]` mm.

We shall proceed as follows:

 1. We compose a function `find_origin` which automatically detects the origin in pixel coordinates of the
   calibration target. This is achieved by using an erosion operation to fill the faint rectangle in the origin,
   and then using OpenCV to detect the location of this blob (origin) of minimum area 18.
 
 1. After this we detect the image points, i.e. smaller circles in a 7x7 grid surrounding the origin and
    store them in an array. We repeat this operation for every calibration image.
   
 1. We construct he object points, i.e. assign the circle grid the expected values in the world coordinate system
    `(x, y, z)` and store them as arrays using the input given to us that the circles on the
    target are evenly spaced by a distance equal to 3 mm.
 
 1. Finally we calibrate the camera.

OpenCV employs a camera model based on the algorithm following Zhang [2000].

Let us start by loading a set of calibration images.

```{code-cell} ipython3
from fluidimage import get_path_image_samples

path = get_path_image_samples() / "TomoPIV" / "calibration" / "cam0"
calib_files = sorted(path.glob("*.tif"))
[path.name for path in calib_files]
```

## Detecting the origin

+++

A typical calibration image looks like:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
from fluidimage.util.util import imread

def imshow(image, ax=plt):
    ax.imshow(image, cmap='gray', vmax=255)

image = imread(str(calib_files[2]))  # z = 0 image
imshow(image)
```

The position of the origin (marked by a rectangle) needs to be detected for detecting the image points consistently.

```{code-cell} ipython3
from fluidimage.util.util import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import reconstruction
import warnings

def rescale(image, scale):
    """Rescale the image intensities"""
    # Scale 16-bit image to 8-bit image
    if scale is None:
        return image
    elif scale == "median":
        scale = np.median(image[image > 5])
    elif scale == "max":
        scale = image.max()

    # print("Rescaling with", scale)
    image = image * (256 / scale)
    return image

def imfill(filename):
    """Fill boundaries in an image. This is used to make the origin easy to detect."""
    image = imread(filename)
    image = rescale(image, "median")
    # Fill the rectangle at the center. Helps to detect the origin
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.max()
    mask = image
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fill the square in the center to mark the origin
        image = reconstruction(seed, mask, method='erosion')

    return image.astype(np.uint8)


test_calib_file = str(calib_files[2])
fig, axes = plt.subplots(1, 2, dpi=100)
imshow(imread(test_calib_file), axes[0])
imshow(imfill(test_calib_file), axes[1])
```

To detect the origin we use `SimpleCircleGrid` class. Although we intend to detect only one point, it works by tweaking the `minArea` parameter. This class will be described in the next section.

```{code-cell} ipython3
from fluidimage.calibration.calib_cv import SimpleCircleGrid
import os

def find_origin(filename):
    params = SimpleCircleGrid.create_default_params()
    params.minArea = 18.

    circle_grid = SimpleCircleGrid(params)

    # Pass the filled image to detect the square
    keypoints = circle_grid.detect_all(
        imfill(str(filename)))
    assert len(keypoints) == 1
    return keypoints[0].pt

for cfile in calib_files:
    print(f"Origin of {cfile.name.rjust(13)} detected at", find_origin(cfile))
```

## Detecting image points as a circle grid

+++

The result is a list of blobs in pixel coordinates, centers in image coordinates.

```{code-cell} ipython3
from fluidimage.calibration.calib_cv import SimpleCircleGrid

params = SimpleCircleGrid.create_default_params()
params
```

There are certain parameters which can be tweaked to detect the circles as needed. For this particular case the defaults are enough.

```{code-cell} ipython3
def construct_image_points(filename, debug=False):
    image = imread(str(filename))
    image = rescale(image, "max")
    origin = find_origin(filename)
    if debug:
        print("Origin =", origin)

    params = SimpleCircleGrid.create_default_params()
    circle_grid = SimpleCircleGrid(params)
    centers = circle_grid.detect_grid(
        image, origin, nx=7, ny=7, ds=50, debug=debug)

    return centers

centers = construct_image_points(calib_files[2], debug=True)
```

## Object Points

+++

The calibrate function requires objectPoints (world coordinates) and imagePoints (image coordinates) of the blobs detected.

```{code-cell} ipython3
from fluidimage.calibration.calib_cv import construct_object_points
```

For example

```{code-cell} ipython3
construct_object_points(nx=3, ny=3, z=-1, ds=3)
```

## Calibration

+++

We now put together all the elements above to calibrate

```{code-cell} ipython3
from pathlib import Path
from tempfile import gettempdir
from fluidimage.calibration.calib_cv import CalibCV


path_output = Path(gettempdir()) / "fluidimage_opencv_calib"


def calibrate_camera(cam="cam0", debug=False):
    path_calib_h5 = path_output / (cam + ".h5")
    calib = CalibCV(path_calib_h5)

    objpoints = []
    imgpoints = []
    zs = []

    path = get_path_image_samples() / "TomoPIV" / "calibration" / cam
    files = sorted(list(path.glob("*.tif")))

    # Populate objpoints, imgpoints and zs
    for i, filename in enumerate(files):
        z = int(filename.name.split("mm_")[0])
        zs.append(z)
        objpoints.append(
            construct_object_points(nx=7, ny=7, z=z, ds=3)
        )
        centers = construct_image_points(str(filename))
        imgpoints.append(centers)

    im_shape = imread(str(filename)).shape[::-1]
    origin = find_origin(str(files[2]))
    return calib.calibrate(imgpoints, objpoints, zs, im_shape, origin, debug)


ret, mtx, dist, rvecs, tvecs = calibrate_camera("cam0", debug=True)
```

```{code-cell} ipython3
from pprint import pformat

def print_horizontally(vecs):
    vecs2 = ['']
    vecs2.extend([v.T for v in vecs])
    return pformat(vecs2)

print(f"""
        Avg. reprojection error = {ret}
        fx, fy = {mtx[0,0]}, {mtx[1,1]}
        cx, cy = {mtx[0,2]}, {mtx[1,2]}
        k1, k2, p1, p2, k3 = {dist.T}
        rotation vectors = {print_horizontally(rvecs)}
        translation vectors = {print_horizontally(tvecs)}
""")
```

## Calibrate all 4 cameras

```{code-cell} ipython3
for i in range(4):
    calibrate_camera(f"cam{i}")
```

```{code-cell} ipython3
list(path_output.glob("*.h5"))
```
