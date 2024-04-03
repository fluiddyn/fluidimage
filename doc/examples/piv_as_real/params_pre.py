#! /usr/bin/env python
"""Contains the parameters for the image preprocessing
======================================================

This can be run in ipython to explore the preprocessing parameters.

To find good parameters, execute the preprocessing topology (see `job_pre.py`)::

  ./job_pre.py 0 2 -how recompute

and use the GUI tool `fluidimviewer` in the input and output directories::

  fluidimviewer ../../../image_samples/Jet/Images.pre &
  fluidimviewer ../../../image_samples/Jet/Images &

You can also try the piv computation with::

  ./try_pre.py &

"""

from params_piv import get_path

from fluidimage.topologies.preproc import TopologyPreproc


def make_params_pre(iexp, savinghow="recompute", postfix_out="pre"):

    # These parameters can depend on the experiment.
    # One can add here a lot of conditions to set good values.
    # DO NOT change the default value if you want to change the value for 1 experiment

    path_images_dir = get_path(iexp)
    assert path_images_dir

    params = TopologyPreproc.create_default_params()
    params.series.path = path_images_dir

    print(f"{path_images_dir}")

    params.tools.sequence = ["rescale_intensity_tanh", "sliding_median"]

    params.tools.rescale_intensity_tanh.enable = False
    params.tools.rescale_intensity_tanh.threshold = None

    params.tools.sliding_median.enable = True
    params.tools.sliding_median.window_size = 10
    params.tools.sliding_median.weight = 0.8

    params.saving.how = savinghow
    params.saving.postfix = postfix_out

    return params


if __name__ == "__main__":
    from args import parse_args

    args = parse_args(doc="", postfix_in=None, postfix_out="pre")
    params = make_params_pre(
        args.exp, savinghow=args.saving_how, postfix_out=args.postfix_out
    )

    print(params)
