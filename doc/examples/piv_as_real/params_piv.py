#! /usr/bin/env python
"""Contains the parameters for the PIV computation
==================================================

This can be run in ipython to explore the PIV parameters.

To find good parameters, try the piv computation with::

  ./try_piv.py &

"""

from pathlib import Path

from fluidimage.piv import Topology

path_here = Path(__file__).absolute().parent


def get_path(iexp):
    """Silly example of get_path function..."""
    return (path_here / "../../../image_samples/Jet/Images").resolve()


def make_params_piv(
    iexp, savinghow="recompute", postfix_in="pre", postfix_out="piv"
):

    # These parameters can depend on the experiment.
    # One can add here a lot of conditions to set good values.
    # DO NOT change the default value if you want to change the value for 1 experiment

    path_images_dir = get_path(iexp)
    assert path_images_dir.exists()

    if postfix_in == "":
        path_in = path_images_dir
    else:
        if not postfix_in.startswith("."):
            postfix_in = "." + postfix_in
        path_in = path_images_dir.with_suffix(postfix_in)

    params = Topology.create_default_params()

    params.series.path = path_in / ("c*.png")
    print(f"{params.series.path = }")

    params.piv0.shape_crop_im0 = 48
    params.piv0.displacement_max = 5

    # for multi peaks search
    params.piv0.nb_peaks_to_search = 2
    params.piv0.particle_radius = 3

    params.mask.strcrop = ":, 50:"

    params.multipass.number = 2
    params.multipass.use_tps = "last"
    params.multipass.smoothing_coef = 5.0
    params.multipass.threshold_tps = 0.1

    params.fix.correl_min = 0.2
    params.fix.threshold_diff_neighbour = 2

    params.saving.how = savinghow
    params.saving.postfix = postfix_out

    return params


if __name__ == "__main__":
    from args import parse_args

    args = parse_args(doc="", postfix_in="pre", postfix_out="piv_try")
    params = make_params_piv(
        args.exp,
        savinghow=args.saving_how,
        postfix_in=args.postfix_in,
        postfix_out=args.postfix_out,
    )
    print(params)
