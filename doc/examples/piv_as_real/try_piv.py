#! /usr/bin/env python
"""Experiment on the parameters of the PIV computation
======================================================

It can be convenient to use this script with ipython --matplotlib

Run it, play with the object `piv` which represents the results, change the
parameters in `params_piv.py` and rerun `try_piv.py`.

Alternatively, run the script with::

  ./try_piv.py &

"""

from importlib import reload

import params_piv

from fluidimage.piv import Work

reload(params_piv)

params = params_piv.make_params_piv(iexp=0)

work = Work(params=params)

piv = work.process_1_serie()

# piv.piv0.display(show_interp=True, scale=0.05, show_error=True)
piv.display(show_interp=False, scale=0.05, show_error=True)
