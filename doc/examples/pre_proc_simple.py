from __future__ import print_function
from fluidimage.pre_proc.base import PreprocBase


params = PreprocBase.create_default_params()

params.preproc.series.path = '../../image_samples/Karman/Images'
params.preproc.series.ind_start = 1

print('Available preprocessing tools: ', params.preproc.tools.available_tools)

params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.filter_size = 20.

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 10.
params.preproc.tools.global_threshold.maxima = 1e3

preproc = PreprocBase(params)
sequence = ['global_threshold', 'sliding_median']  # Optional
preproc(sequence)
