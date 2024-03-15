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

# PIV computation

In this tutorial, we will see how to compute PIV (Particle Image Velocimetry) fields and
how to load the results. In Fluidimage, we have the concept of "works" and "topologies"
of works. Computing the PIV from 2 images is a "work" (defined in
{class}`fluidimage.piv.Work`). The PIV topology ({class}`fluidimage.piv.Topology`) uses
in the background the PIV work and also takes care of reading the images, creating the
couples of images and saving the results.

A Fluidimage topology can be executed with different executors, which usually runs the
different tasks in parallel.

+++

## Finding good parameters

In order to look for good parameters, we usually compute only PIV fields just with the
PIV work {class}`fluidimage.piv.Work` (i.e. without topology).

```{code-cell} ipython3
---
tags: [remove-output]
---
from fluidimage import get_path_image_samples
from fluidimage.piv import Work
```

We use the class function `create_default_params` to create an object containing the
parameters.

```{code-cell} ipython3
params = Work.create_default_params()
```

The representation of this object is useful. In Ipython, just write:

```{code-cell} ipython3
params
```

We see a representation of the default parameters. The documention can be printed with
`_print_doc`, for example for `params.multipass`:

```{code-cell} ipython3
params.multipass._print_doc()
```

We can of course modify these parameters. An error will be raised if we accidentally try
to modify a non existing parameter. We at least need to give information about where are
the input images:

```{code-cell} ipython3
path_src = get_path_image_samples() / "Karman/Images"
params.series.path = str(path_src)

params.piv0.shape_crop_im0 = 64
params.multipass.number = 2
params.multipass.use_tps = False
```

After instantiating the `Work` class,

```{code-cell} ipython3
work = Work(params)
```

we can use it to compute one PIV field with the function
{func}`fluidimage.works.BaseWorkFromSerie.process_1_serie`:

```{code-cell} ipython3
result = work.process_1_serie()
```

```{code-cell} ipython3
type(result)
```

This object of the class {class}`fluidimage.data_objects.piv.MultipassPIVResults`
contains all the final and intermediate results of the PIV computation.

A PIV computation is usually made in different passes (most of the times, 2 or 3). The
pass `n` uses the results of the pass `n-1`.

```{code-cell} ipython3
[s for s in dir(result) if not s.startswith('_')]
```

```{code-cell} ipython3
piv0, piv1 = result.passes
assert piv0 is result.piv0
assert piv1 is result.piv1
type(piv1)
```

Let's see what are the attributes for one pass (see
{class}`fluidimage.data_objects.piv.HeavyPIVResults`). First the first pass:

```{code-cell} ipython3
[s for s in dir(piv0) if not s.startswith('_')]
```

and the second pass:

```{code-cell} ipython3
[s for s in dir(piv1) if not s.startswith('_')]
```

The main raw results of a pass are `deltaxs`, `deltays` (the displacements) and `xs` and
`yx` (the locations of the vectors, which depend of the images).

```{code-cell} ipython3
assert piv0.deltaxs.shape == piv0.deltays.shape == piv0.xs.shape == piv0.ys.shape
piv0.xs.shape
```

```{code-cell} ipython3
assert piv1.deltaxs.shape == piv1.deltays.shape == piv1.xs.shape == piv1.ys.shape
piv1.xs.shape
```

`piv0.deltaxs_approx` is an interpolation on a grid used for the next pass (saved in
`piv0.ixvecs_approx`)

```{code-cell} ipython3
assert piv0.deltaxs_approx.shape == piv0.ixvecs_approx.shape == piv0.deltays_approx.shape == piv0.iyvecs_approx.shape

import numpy as np
# there is also a type change between these 2 arrays
assert np.allclose(
    np.round(piv0.deltaxs_approx).astype("int32"),  piv1.deltaxs_input
)
```

For the last pass, there is also the corresponding `piv1.deltaxs_final` and
`piv1.deltays_final`, which are computed on the final grid (`piv1.ixvecs_final` and
`piv1.iyvecs_final`).

```{code-cell} ipython3
assert piv1.deltaxs_final.shape == piv1.ixvecs_final.shape == piv1.deltays_final.shape == piv1.iyvecs_final.shape
```

```{code-cell} ipython3
help(result.display)
```

```{code-cell} ipython3
result.display(show_correl=False, hist=True);
```

```{code-cell} ipython3
result.display(show_interp=True, show_correl=False, show_error=False);
```

We could improve the results, but at least they seem coherent, so we can use these simple
parameters with the PIV topology (usually to compute a lot of PIV fields).

+++

## Instantiate the topology and launch the computation

Let's first import what will be useful for the computation, in particular the class
{class}`fluidimage.piv.Topology`.

```{code-cell} ipython3
import os

from fluidimage.piv import Topology
```

We use the class function `create_default_params` to create an object containing the
parameters.

```{code-cell} ipython3
params = Topology.create_default_params()
```

The parameters for the PIV topology are nearly the same than those of the PIV work. One
noticable difference is the addition of `params.saving`, because a topology saves its
results. One can use `_print_as_code` to print parameters as in Python code (useful for
copy/pasting).

```{code-cell} ipython3
params.saving._print_as_code()
```

```{code-cell} ipython3
params.saving._print_doc()
```

```{code-cell} ipython3
path_src = get_path_image_samples() / "Karman/Images"
params.series.path = str(path_src)

params.piv0.shape_crop_im0 = 64
params.multipass.number = 2
params.multipass.use_tps = False

params.saving.how = 'recompute'
params.saving.postfix = "doc_piv_ipynb"
```

In order to run the PIV computation, we have to instanciate an object of the class
{class}`fluidimage.piv.Topology`.

```{code-cell} ipython3
topology = Topology(params)
```

We will then launch the computation by running the function `topology.compute`. For this
tutorial, we use a sequential executor to get a simpler logging.

However, other Fluidimage topologies usually launch computations in parallel so that it
is mandatory to set the environment variable `OMP_NUM_THREADS` to `"1"`.

```{code-cell} ipython3
os.environ["OMP_NUM_THREADS"] = "1"
```

Let's go!

```{code-cell} ipython3
topology.compute(sequential=True)
```

```{code-cell} ipython3
path_src
```

```{code-cell} ipython3
topology.path_dir_result
```

## Analyzing the computation

```{code-cell} ipython3
from fluidimage.topologies.log import LogTopology
log = LogTopology(topology.path_dir_result)
```

```{code-cell} ipython3
log.durations
```

```{code-cell} ipython3
log.plot_durations()
```

```{code-cell} ipython3
log.plot_memory()
```

```{code-cell} ipython3
log.plot_nb_workers()
```

## Loading the output files

```{code-cell} ipython3
os.chdir(topology.path_dir_result)
```

```{code-cell} ipython3
from fluidimage import create_object_from_file
```

```{code-cell} ipython3
o = create_object_from_file('piv_01-02.h5')
```

```{code-cell} ipython3
[s for s in dir(o) if not s.startswith('_')]
```

```{code-cell} ipython3
[s for s in dir(o.piv1) if not s.startswith('_')]
```

```{code-cell} ipython3
o.display();
```

```{code-cell} ipython3
o.piv0.display(
    show_interp=False, scale=0.1, show_error=False, pourcent_histo=99, hist=False
);
```

The output PIV files are just hdf5 files. If you just want to load the final velocity
field, do it manually with h5py.

```{code-cell} ipython3
import h5py
with h5py.File("piv_01-02.h5", "r") as file:
    deltaxs_final = file["piv1/deltaxs_final"][:]
```

```{raw-cell}
Moreover, the final result of this PIV computation can also be manipulated and visualized using {class}`fluidimage.postproc.vector_field.VectorFieldOnGrid`:
```

```{code-cell} ipython3
from fluidimage.postproc.vector_field import VectorFieldOnGrid

field = VectorFieldOnGrid.from_file("piv_01-02.h5")
```

```{code-cell} ipython3
field.display(scale=0.1);
```

```{code-cell} ipython3
field.gaussian_filter(sigma=1).display(scale=0.1);
```

```{code-cell} ipython3
from shutil import rmtree
rmtree(topology.path_dir_result, ignore_errors=True)
```
