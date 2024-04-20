# Investigating a leak in fluidimage

There was a leak in {func}`fluidimage.calcul.subpix.compute_subpix_2d_gaussian2`,
which is a pythranized function.

A simple reproducer is

```python

# pythran export simpler_leak(float32[:, :], int, int)
# pythran export simpler_no_leak(float32[:, :], int, int)

def simpler_leak(correl, ix, iy):
    # returning this view leads to a leak!
    correl_crop = correl[iy - 1 : iy + 2, ix - 1 : ix + 2]
    return correl_crop

def simpler_no_leak(correl, ix, iy):
    correl_crop = np.ascontiguousarray(correl[iy - 1 : iy + 2, ix - 1 : ix + 2])
    return correl_crop
```
