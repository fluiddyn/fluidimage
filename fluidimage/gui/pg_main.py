#!/usr/bin/env python
"""PyQtGraph app which opens a single image or multiple images
in separate docks / as a slideshow.

Examples
--------
$ fluidimviewer-pg im1.png im2.png
$ fluidimviewer-pg im???a.png
$ fluidimviewer-pg -s im???a.png

Dock mode
---------
* Displays histogram of image data with movable region defining the dark/light
  levels
* Region of interest (ROI) and embedded plot for measuring image values
  across frames
* Image normalization / background subtraction

Slideshow mode: Keyboard interaction
------------------------------------
* left/right arrows step forward/backward 1 frame when pressed, seek at 20fps
  when held.
* up/down arrows seek at 100fps
* pgup/pgdn seek at 1000fps
* home/end seek immediately to the first/last frame
* space begins playing frames. If time values (in seconds) are given for each
  frame, then playback is in realtime.

"""
import argparse
import os
import sys

from fluidimage import config_logging
from fluidimage.gui.pg_wrapper import PGWrapper


def dock(args, for_testing):
    pg = PGWrapper()
    for arg in args:
        title = os.path.basename(arg)
        pg._add_dock(title, size=(500, 500), position="right")
        pg.view(arg, title)

    if not for_testing:
        pg.show()
    return pg


def slideshow(args, for_testing):
    title = "FluidImage {} to {}".format(
        os.path.basename(args[0]), os.path.basename(args[-1])
    )
    pg = PGWrapper(title=title)

    pg.view(args, title)
    if not for_testing:
        pg.show()
    return pg


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("images", help="Path to the image or pattern", nargs="+")

    parser.add_argument(
        "-s",
        "--slideshow",
        help="View multiple images as a slideshow. If not open them in docks.",
        action="store_true",
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args(args)


def main(args=None, for_testing=False):
    """Parse arguments and execute `fluidimviewer-pg`."""
    if args is None:
        args = parse_args()

    if args.verbose:
        config_logging("debug")
    else:
        config_logging()

    if args.slideshow:
        return slideshow(args.images, for_testing)
    else:
        return dock(args.images, for_testing)


if __name__ == "__main__":
    main()
