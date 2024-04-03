#! /usr/bin/env python
"""
Job computing all PIV fields
============================

To be launched in the terminal::

  ./job_piv.py 0 --nb-cores 2

see also the help::

  ./job_piv.py -h

"""

import params_piv
from job_pre import check_nb_cores

from fluidimage.piv import Topology


def main(args):
    params = params_piv.make_params_piv(
        args.exp,
        savinghow=args.saving_how,
        postfix_in=args.postfix_in,
        postfix_out=args.postfix_out,
    )

    check_nb_cores(args)

    topology = Topology(params, nb_max_workers=int(args.nb_cores))
    topology.compute(sequential=args.seq)


if __name__ == "__main__":
    from args import parse_args

    args = parse_args(doc=__doc__, postfix_in="pre", postfix_out="piv")
    print(args)
    main(args)
