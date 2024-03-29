"""Contains the parameters for the image preprocessing (params_pre.py)
======================================================================

This can be run in ipython to explore the preprocessing parameters.

To find good parameters, execute the preprocessing topology (see `job_pre.py`)::

  ./job_pre.py 0 2 -how recompute

and use the GUI tool `fluidimviewer` in the input and output directories::

  fluidimviewer ../../../image_samples/Jet/Images.pre &
  fluidimviewer ../../../image_samples/Jet/Images &

You can also try the piv computation with::

  ./try_piv.py &

"""

from copy import deepcopy
from glob import glob

from params_piv import get_path

from fluidimage.topologies.preproc import TopologyPreproc


def make_params_pre(iexp, savinghow="recompute", postfix_out="pre"):
    path = get_path(iexp)

    params = TopologyPreproc.create_default_params()
    params.series.path = path

    print("path", path)
    str_glob = path + "/c*.png"
    paths = glob(str_glob)
    if len(paths) == 0:
        raise ValueError('No images detected from the string "' + str_glob + '"')

    pathim = paths[0]
    double_frame = pathim.endswith("a.png") or pathim.endswith("b.png")
    if double_frame:
        params.series.str_subset = "i:i+1, 0"
    else:
        params.series.str_subset = "i:i+1"

    params.series.ind_start = 60
    params.series.ind_stop = 62

    params.tools.sequence = ["rescale_intensity_tanh", "sliding_median"]

    params.tools.rescale_intensity_tanh.enable = False
    params.tools.rescale_intensity_tanh.threshold = None

    params.tools.sliding_median.enable = True
    params.tools.sliding_median.window_size = 10
    params.tools.sliding_median.weight = 0.8

    params.saving.how = savinghow
    params.saving.postfix = postfix_out

    if double_frame:
        # for 'b.png' images
        params2 = deepcopy(params)
        params2.preproc.series.str_subset = params.series.str_subset[:-1] + "1"
        return [params, params2]
    else:
        return [params]


if __name__ == "__main__":
    from fluidcoriolis.milestone.args_piv import parse_args

    args = parse_args(doc="", postfix_in=None, postfix_out="pre")
    list_params = make_params_pre(
        args.exp, savinghow=args.saving_how, postfix_out=args.postfix_out
    )

    params = list_params[0]
    try:
        params2 = list_params[1]
    except IndexError:
        pass

    print(params)
