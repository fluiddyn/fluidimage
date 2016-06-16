from __future__ import print_function
from fluidimage.topologies.pre_proc import TopologyPreproc


params = TopologyPreproc.create_default_params()

params.preproc.series.path = '../../image_samples/Karman/Images'
params.preproc.series.ind_start = 1

print('Available preprocessing tools: ', params.preproc.tools.available_tools)

params.preproc.tools.sequence = ['sliding_median', 'global_threshold']
params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 10

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 0.
params.preproc.tools.global_threshold.maxima = 255.

topology = TopologyPreproc(params)

topology.compute(sequential=False)
