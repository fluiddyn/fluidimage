
from __future__ import print_function

from fluidimage.preproc.base import PreprocBase

params = PreprocBase.create_default_params()

params.preproc.series.path = '../../image_samples/Karman/Images'
print('Available preprocessing tools: ', params.preproc.tools.available_tools)

params.preproc.tools.sequence = ['sliding_median', 'global_threshold']
params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 25

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 0.
params.preproc.tools.global_threshold.maxima = 255.

preproc = PreprocBase(params)
preproc()

preproc.display(1, hist=False)
