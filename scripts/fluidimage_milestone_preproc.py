#!/usr/bin/env python
'''
Preprocess a particular experiment, camera and level / or multiple levels.

Examples
--------
All levels
    $ ./fluidimage_milestone_preproc.py 35 PCO_side

All levels with a particular pattern
    $ ./fluidimage_milestone_preproc.py 35 PCO_side -p level?? -vv

A single level
    $ ./fluidimage_milestone_preproc.py 35 PCO_side -l level00

Note
----
For help on how to launch the script, execute
    $ ./fluidimage_milestone_preproc.py -h

'''
from __future__ import print_function

import argparse
from fluidcoriolis.milestone import path_exp
from fluidimage import config_logging
from fluidimage.util.paramlist import ParamListPreproc


def params_PCO(params, frames, letter=None):
    '''Parameters to preprocess PCO images'''

    if frames == 1:
        params.preproc.series.strcouple = 'i:i+23'
    elif frames == 2:
        if letter == 'a':
            params.preproc.series.strcouple = 'i:i+10,0'
        if letter == 'b':
            params.preproc.series.strcouple = 'i:i+10,1'

    params.preproc.saving.postfix = 'fsback'
    params.preproc.saving.how = 'complete'
    params.preproc.saving.format = 'img'

    params.preproc.tools.sequence = ['temporal_percentile',
                                     'equalize_hist_adapt']

    params.preproc.tools.temporal_percentile.enable = True
    params.preproc.tools.temporal_percentile.percentile = 10.

    params.preproc.tools.equalize_hist_adapt.enable = True
    params.preproc.tools.equalize_hist_adapt.window_shape = (200, 200)
    params.preproc.tools.equalize_hist_adapt.nbins = 65535
    return params


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'exp', help='index of the experiment (int)', type=int)
    parser.add_argument(
        'camera', help='name of the camera (str)', type=str)
    parser.add_argument(
        '-l', '--level',
        help='name of the level subdirectory (str).\
                if unspecified preprocess all levels',
        type=str)
    parser.add_argument(
        '-p', '--pattern',
        help='glob expression of the level subdirectory (str).',
        type=str)
    parser.add_argument(
        '-s', '--seq',
        help='launch topologies sequentially',
        action='store_true')
    parser.add_argument(
        '-t', '--test',
        help='test mode. launches one work',
        action='store_true')
    parser.add_argument(
        '-v', '--verbose',
        help='verbose mode.',
        action='count')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.verbose == 2:
        print('Logger in INFO mode')
        config_logging('info')
    elif args.verbose >= 3:
        print('Logger in DEBUG mode')
        config_logging('debug')

    camera_params = {'PCO_top': params_PCO,
                     'PCO_bottom': params_PCO,
                     'PCO_side': params_PCO}

    param_list = ParamListPreproc(camera_specific_params=camera_params,
                                  path_list=path_exp)

    param_list.init_directory(args.exp, args.camera)

    if args.level is not None:
        param_list.fill_params(args.level)
    else:
        for level in param_list.get_levels(args.pattern):
            param_list.fill_params(level)

    if not args.test:
        param_list.launch_topologies(args.seq, args.verbose)
    else:
        raise NotImplementedError
