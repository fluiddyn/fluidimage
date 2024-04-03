#! /usr/bin/env python
"""
Job computing preprocessing
===========================

To be launched in the terminal::

  ./job_pre.py 0 --nb-cores 2

or (to force the processing of already computed images)::

  ./job_pre.py 0 --nb-cores 2 -how recompute

see also the help::

  ./job_pre.py -h

"""

import sys
from importlib import reload

import params_pre

from fluidimage.preproc import Topology

reload(params_pre)


def check_nb_cores(args):
    if args.nb_cores is None:
        print(
            "Bad argument: nb_cores is None. Specify with --nb-cores",
            file=sys.stderr,
        )
        sys.exit(1)


def main(args):
    params = params_pre.make_params_pre(
        args.exp, savinghow=args.saving_how, postfix_out=args.postfix_out
    )

    check_nb_cores(args)

    topology = Topology(params, nb_max_workers=int(args.nb_cores))
    topology.compute(sequential=args.seq)


if __name__ == "__main__":
    from args import parse_args

    args = parse_args(doc=__doc__, postfix_in=None, postfix_out="pre")
    print(args)
    main(args)

    print("\nend of job_pre.py")
