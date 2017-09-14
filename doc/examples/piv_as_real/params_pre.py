"""Contains the parameters for the image preprocessing (params_pre.py)
======================================================================

This can be run in ipython to explore the preprocessing parameters.

To find good parameters, execute the preprocessing topology (see `job_pre.py`)
and use the GUI tool `fluidimviewer` in the input and output directories.

"""

from copy import deepcopy
from glob import iglob

from fluidimage.topologies.preproc import TopologyPreproc

from params_piv import get_path


def make_params_pre(iexp, savinghow='recompute',
                    postfix_out='pre'):
    path = get_path(iexp)

    params = TopologyPreproc.create_default_params()
    params.preproc.series.path = path

    it = iglob(path + '/c*.png')
    pathim = next(it)
    double_frame = pathim.endswith('a.png') or pathim.endswith('b.png')
    if double_frame:
        params.preproc.series.strcouple = 'i:i+1, 0'
    else:
        params.preproc.series.strcouple = 'i:i+1'

    params.preproc.series.ind_start = 60
    params.preproc.series.ind_stop = 62

    params.preproc.tools.sequence = ['rescale_intensity_tanh', 'sliding_median']

    params.preproc.tools.rescale_intensity_tanh.enable = False
    params.preproc.tools.rescale_intensity_tanh.threshold = None

    params.preproc.tools.sliding_median.enable = True
    params.preproc.tools.sliding_median.window_size = 10
    params.preproc.tools.sliding_median.weight = 0.8

    params.preproc.saving.how = savinghow
    params.preproc.saving.postfix = postfix_out

    if double_frame:
        params2 = deepcopy(params)
        params2.preproc.series.strcouple = \
            params.preproc.series.strcouple[:-1] + '1'
        return [params, params2]
    else:
        return [params]

if __name__ == '__main__':
    from fluidcoriolis.milestone.args_piv import parse_args
    args = parse_args(
        doc='', postfix_in=None, postfix_out='pre')
    list_params = make_params_pre(
        args.exp, savinghow=args.saving_how,
        postfix_out=args.postfix_out)

    params = list_params[0]
    try:
        params2 = list_params[1]
    except IndexError:
        pass

    print(params)
