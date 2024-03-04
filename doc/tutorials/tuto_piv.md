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

## Launching the topology of computation for PIV

```{code-cell} ipython3
import os
os.environ["OMP_NUM_THREADS"] = "1"
```

```{code-cell} ipython3
from fluidimage import get_path_image_samples
```

```{code-cell} ipython3
path_src = get_path_image_samples() / "Karman/Images"
postfix = "doc_piv_ipynb"
```

We first import the class {class}`fluidimage.topologies.piv.TopologyPIV`.

```{code-cell} ipython3
from fluidimage.piv import Topology
```

We use a class function to create an object containing the parameters.

```{code-cell} ipython3
params = Topology.create_default_params()
```

The representation of this object is useful. In Ipython, just do:

```{code-cell} ipython3
params
```

We here see a representation of the default parameters. Some elements have a `_doc` attribute. For example:

```{code-cell} ipython3
params.multipass._print_doc()
```

We can of course modify these parameters. An error will be raised if we accidentally try to modify a non existing parameter. We at least need to give information about where are the input images:

```{code-cell} ipython3
params.series.path = str(path_src)

params.piv0.shape_crop_im0 = 64
params.multipass.number = 2
params.multipass.use_tps = False

params.saving.how = 'recompute'
params.saving.postfix = postfix
```

In order to run the PIV computation, we have to instanciate an object of the class {class}`fluidimage.topologies.piv.TopologyPIV`.

```{code-cell} ipython3
topology = Topology(params)
```

We then launch the computation by running the function `topology.compute`. We use a sequential executor to get a simpler logging.

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
print([s for s in dir(o) if not s.startswith('_')])
```

```{code-cell} ipython3
print([s for s in dir(o.piv1) if not s.startswith('_')])
```

```{code-cell} ipython3
o.display()
```

```{code-cell} ipython3
o.piv0.display(show_interp=False, scale=0.1, show_error=False, pourcent_histo=99, hist=False, show_correl=True)
```

The output PIV files are just hdf5 files. If you just want to load the final velocity field, do it manually with h5py.

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
field.display(scale=0.1)
```

```{code-cell} ipython3
field.gaussian_filter(sigma=1).display(scale=0.1)
```

```{code-cell} ipython3
from shutil import rmtree
rmtree(topology.path_dir_result, ignore_errors=True)
```

```{code-cell} ipython3
os.chdir(path_src)
```
