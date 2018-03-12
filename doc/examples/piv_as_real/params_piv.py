"""Contains the parameters for the PIV computation (params_piv.py)
==================================================================

This can be run in ipython to explore the PIV parameters.

To find good parameters, try the piv computation with::

  ./try_piv.py &

"""
from glob import glob

from fluidimage.topologies.piv import TopologyPIV


def get_path(iexp):
    """Silly example of get_path function..."""
    return '../../../image_samples/Jet/Images'


def make_params_piv(iexp, savinghow='recompute',
                    postfix_in='pre', postfix_out='piv'):

    path = get_path(iexp)

    if postfix_in is not None and postfix_in != '':
        path += '.' + postfix_in

    params = TopologyPIV.create_default_params()
    params.series.path = path

    print('path', path)
    str_glob = path + '/c*.png'
    paths = glob(str_glob)
    if len(paths) == 0:
        raise ValueError(
            'No images detected from the string "' + str_glob + '"')

    pathim = paths[0]
    if pathim.endswith('a.png') or pathim.endswith('b.png'):
        params.series.strcouple = 'i, 0:2'
    else:
        params.series.strcouple = 'i:i+2'
    params.series.ind_start = 60
    params.series.ind_stop = None

    params.piv0.shape_crop_im0 = 48
    params.piv0.method_correl = 'fftw'
    params.piv0.displacement_max = 3

    params.mask.strcrop = ':, 50:'

    params.multipass.number = 2
    params.multipass.use_tps = 'last'
    params.multipass.smoothing_coef = 10.
    params.multipass.threshold_tps = 0.1

    params.fix.correl_min = 0.07
    params.fix.threshold_diff_neighbour = 1

    params.saving.how = savinghow
    params.saving.postfix = postfix_out

    return params

if __name__ == '__main__':
    from args import parse_args
    args = parse_args(
        doc='', postfix_in='pre', postfix_out='piv_try')
    params = make_params_piv(
        args.exp, savinghow=args.saving_how,
        postfix_in=args.postfix_in, postfix_out=args.postfix_out)
    print(params)
