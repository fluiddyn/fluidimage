#!/usr/bin/env python
'''Opens multiple images as a slideshow.
Usage: fluidimage_slideshow.py [image1 image2] [pattern]

Examples
--------
$ fluidimage_slideshow.py im1.png im2.png
$ fluidimage_slideshow.py im???a.png

Keyboard interaction
--------------------
* left/right arrows step forward/backward 1 frame when pressed, seek at 20fps when held.
* up/down arrows seek at 100fps
* pgup/pgdn seek at 1000fps
* home/end seek immediately to the first/last frame
* space begins playing frames. If time values (in seconds) are given for each frame, then playback is in realtime.

'''
from __future__ import print_function
import sys
import os
from fluidimage.gui.pg_wrapper import PGWrapper
from fluidimage import config_logging

config_logging('debug')

args = sys.argv[1:]
if len(args) == 0:
    print(__doc__)
    sys.exit()

title = 'FluidImage {} to {}'.format(
        os.path.basename(args[0]), os.path.basename(args[-1]))
pg = PGWrapper(title=title)

pg.view(args, title)
pg.show()
