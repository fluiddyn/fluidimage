# How to find good PIV parameters?

Fluidimage is designed to process several images in parallel with asynchronous
"topologies". However, the asynchronous PIV topology is not very convenient to
look for the best parameters for a PIV computation. It is better to use the PIV
"work" directly as in these examples:

```{literalinclude} piv_try_params.py
```

The parameters in `params.series` are used to define a
{class}`fluiddyn.util.serieofarrays.SeriesOfArrays`
and to select one serie (which represents here a couple of images). It is also
what is done internally in the PIV topology. Have a look at
[our tutorial](https://fluiddyn.readthedocs.io/en/latest/ipynb/tuto_serieofarrays.html)
to discover how to use this powerful tool!

This other example includes a simple image-to-image preprocessing.

```{literalinclude} piv_try_params_with_im2im_preproc.py
```
