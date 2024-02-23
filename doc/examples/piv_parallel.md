# PIV computation in parallel with `TopologyPIV`

This minimal example presents how to carry out a simple PIV computation.  See
also the documentation of the class
{class}`fluidimage.topologies.piv.TopologyPIV` and the work defined in the
subpackage {mod}`fluidimage.works.piv`.

```{literalinclude} piv_parallel.py
```

We now show a similar example but with a simple preprocessing (using a function
`im2im`):

```{literalinclude} piv_parallel_im2im.py
```

The file `my_example_im2im.py` should be importable (for example in the same
directory than `piv_parallel_im2im.py`)

```{literalinclude} my_example_im2im.py
```

Same thing but the preprocessing is done with a class `Im2Im`

```{literalinclude} piv_parallel_im2im_class.py
```

The file `my_example_im2im_class.py` should be importable (for example in the
same directory than `piv_parallel_im2im_class.py`)

```{literalinclude} my_example_im2im_class.py
```
