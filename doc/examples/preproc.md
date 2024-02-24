# Preprocessing images

There are two main methods to preprocess images with Fluidimage:

1. with user-defined preprocessing functions or classes, or
2. with pre-defined preprocessing classes.

For both methods, we use a `Work` class to try parameters and a `Topology`
class to apply this work in parallel on a large serie of images.
See the [](../overview_orga_package.md) for an explanation about this terminology.

## 1. User-defined preprocessing

Let's assume that you have a function or a class which gets an image as
argument and returns a new image. If it is importable, you can use Fluidimage
to first investigate which are the better parameters for your case:

```{literalinclude} im2im_try_params.py
```

and then process a large serie of images in parallel with the class
{class}`fluidimage.image2image.Topology`.

```{literalinclude} im2im_parallel.py
```

## 2. Pre-defined preprocessing classes

Fluidimage also provides pre-defined preprocessing classes to apply many
standard preprocessing to images.

### Preprocessing one serie

To find the good parameter, you can use the class
{class}`fluidimage.preproc.Work` (see also
{mod}`fluidimage.preproc`).

```{literalinclude} preproc_try_params.py
```

### Preprocessing large series of images

To apply the preprocessing to a large serie of images in parallel, use
{class}`fluidimage.preproc.Topology`.

```{literalinclude} preproc_parallel.py
```
