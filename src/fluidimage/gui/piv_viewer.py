"""PIV viewer

Coded with matplotlib GUI!

.. autoclass:: VectorFieldsViewer
   :members:
   :private-members:

"""

import argparse
import os
from pathlib import Path


class VectorFieldsViewer:
    """A simple vector field viewer."""

    @classmethod
    def parse_args(cls):

        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "path",
            help="Path file or directory.",
            type=str,
            nargs="?",
            default=os.getcwd(),
        )
        parser.add_argument(
            "-v", "--verbose", help="verbose mode", action="count"
        )

        return parser.parse_args()

    def __init__(self, args):
        pass


def main():
    args = VectorFieldsViewer.parse_args()
    VectorFieldsViewer(args)
