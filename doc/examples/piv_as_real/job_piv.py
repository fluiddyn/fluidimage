#! /usr/bin/env python
"""
Job computing all PIV fields (job_piv.py)
=========================================

To be launched in the terminal::

  ./job_piv.py 0 2

see also the help::

  ./job_piv.py -h

"""

from __future__ import print_function

import params_piv
from fluidimage.topologies.piv import TopologyPIV


def main(args):

    params = params_piv.make_params_piv(
        args.exp, savinghow=args.saving_how,
        postfix_in=args.postfix_in, postfix_out=args.postfix_out)

    topology = TopologyPIV(params, nb_max_workers=int(args.nb_cores))
    topology.compute(sequential=args.seq)

if __name__ == '__main__':

    from fluidcoriolis.milestone.args_piv import parse_args
    args = parse_args(
        doc=__doc__, postfix_in='pre', postfix_out='piv')
    print(args)
    main(args)
