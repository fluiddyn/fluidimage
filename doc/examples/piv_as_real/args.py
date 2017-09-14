"""module to define command line tools with argparse (args.py)
==============================================================

It is used in other scripts in this directory.

"""
import argparse


def make_parser(doc='', postfix_in='pre', postfix_out='piv_coarse'):
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'exp', help='index of the experiment (int)', type=int)
    parser.add_argument(
        'nb_cores', help='nb_cores', type=int)
    parser.add_argument(
        '-s', '--seq', help='launch topologies sequentially',
        action='store_true')
    parser.add_argument('-v', '--verbose', help='verbose mode', action='count')
    parser.add_argument(
        '-how', '--saving_how', type=str,
        help='Saving mode ("ask", "new_dir", "complete" or "recompute")',
        default='complete')
    parser.add_argument(
        '-in', '--postfix_in', type=str, help='postfix input',
        default=postfix_in)
    parser.add_argument(
        '-out', '--postfix_out', type=str, help='postfix output',
        default=postfix_out)
    return parser


def parse_args(doc='', postfix_in='pre', postfix_out='piv_coarse'):
    parser = make_parser(doc, postfix_in, postfix_out)
    args = parser.parse_args()
    return args
