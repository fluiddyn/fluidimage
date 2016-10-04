#!/usr/bin/env python
'''Opens a single image or multiple images in seperate docks.
Usage: fluidimage_view.py [image1 image2] [pattern]

Examples
--------
$ fluidimage_view.py im1.png im2.png
$ fluidimage_view.py im???a.png

Features
--------
* Displays histogram of image data with movable region defining the dark/light
  levels
* ROI and embedded plot for measuring image values across frames
* Image normalization / background subtraction

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

pg = PGWrapper()
for arg in args:
    title = os.path.basename(arg)
    pg._add_dock(title, size=(500, 500), position='right')
    pg.view(arg, title)

pg.show()
