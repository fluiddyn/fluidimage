#! /usr/bin/env python
"""Experiment on the parameters of the preproc computation
==========================================================

It can be convenient to use this script with ipython --matplotlib

Run it, change the parameters in `params_pre.py` and rerun `try_pre.py`.

Alternatively, run the script with::

  ./try_pre.py &

"""

from importlib import reload

import params_pre

from fluidimage.preproc import Work

reload(params_pre)

params = params_pre.make_params_pre(iexp=0)

work = Work(params=params)

work.display()
