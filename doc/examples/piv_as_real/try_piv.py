#! /usr/bin/env python
"""Experiment on the parameters of the PIV computation (try_piv.py)
===================================================================

It can be convenient to use this script with ipython --matplotlib

Run it, play with the object `piv` which represents the results, change the
parameters in `params_piv.py` and rerun `try_piv.py`.

Alternatively, run the script with::

  ./try_piv.py &

"""

import params_piv

from fluidimage import SeriesOfArrays
from fluidimage.piv import WorkPIV

try:
    reload
except NameError:
    from importlib import reload


reload(params_piv)

iexp = 0

params = params_piv.make_params_piv(iexp)

work = WorkPIV(params=params)

# c060a.png and c060b.png
piv = work.process_1_serie(params.series.ind_start)

# piv.piv0.display(show_interp=True, scale=0.05, show_error=True)

piv.display(show_interp=False, scale=0.05, show_error=True)
