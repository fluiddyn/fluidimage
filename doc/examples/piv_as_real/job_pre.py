#! /usr/bin/env python
"""
Job computing preprocessing (job_pre.py)
========================================

To be launched in the terminal::

  ./job_pre.py 0 2

or (to force the processing of already computed images)::

  ./job_pre.py 0 2 -how recompute

see also the help::

  ./job_pre.py -h

"""

from __future__ import print_function

import params_pre
from fluidimage.topologies.preproc import TopologyPreproc

try:
    reload
except NameError:
    from importlib import reload

reload(params_pre)


def main(args):

    list_params = params_pre.make_params_pre(
        args.exp, savinghow=args.saving_how, postfix_out=args.postfix_out)

    for i, params in enumerate(list_params):
        print(f'\nTopology for params {i}')
        topology = TopologyPreproc(params, nb_max_workers=int(args.nb_cores))
        topology.compute(sequential=args.seq)

if __name__ == '__main__':

    from args import parse_args
    args = parse_args(
        doc=__doc__, postfix_in=None, postfix_out='pre')
    print(args)
    main(args)

    print('\nend of job_pre.py')
