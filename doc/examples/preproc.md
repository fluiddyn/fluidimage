# Preprocessing images

There are two main methods to preprocess images with Fluidimage:

1. with user-defined preprocessing functions or classes, or
2. with pre-defined preprocessing classes.

## 1. User-defined preprocessing

Let's assume that you have a function or a class which gets an image as
argument and returns a new image. If it is importable, you can use Fluidimage
to process a large serie of images in parallel with the class
{class}`fluidimage.topologies.image2image.TopologyImage2Image`.

```{literalinclude} preproc_userdefined.py
```

## 2. Pre-defined preprocessing classes

Fluidimage also provides pre-defined preprocessing classes to apply many
standard preprocessing to images.

### Preprocessing one serie

To find the good parameter, you can use the class
{class}`fluidimage.preproc.base.PreprocBase` (see also
{mod}`fluidimage.preproc`).

```{literalinclude} preproc_try_params.py
```

### Preprocessing large series of images

To apply the preprocessing to a large serie of images in parallel, use
{class}`fluidimage.topologies.preproc.TopologyPreproc`.

```{literalinclude} preproc_with_topology.py
```
