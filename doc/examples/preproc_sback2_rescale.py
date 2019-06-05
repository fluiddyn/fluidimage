from __future__ import print_function

from fluidimage.topologies.preproc import TopologyPreproc

params = TopologyPreproc.create_default_params()

params.preproc.series.path = (
    ('$HOME/useful/project/16MILESTONE/Data/' +
     'Exp21_2016-06-22_N0.8_L6.0_V0.08_piv3d/PCO_top/level2.tfilter'))

params.preproc.series.strcouple = 'i:i+1'
params.preproc.series.ind_start = 1200
params.preproc.series.ind_stop = 1224

params.preproc.saving.postfix = 'rescale'
params.preproc.saving.format = 'img'

params.preproc.tools.sequence = ['equalize_hist_global', 'equalize_hist_adapt',
                                 'equalize_hist_local', 'rescale_intensity']

params.preproc.tools.equalize_hist_global.enable = False
params.preproc.tools.equalize_hist_global.nbins = 500

params.preproc.tools.equalize_hist_adapt.enable = True
params.preproc.tools.equalize_hist_adapt.window_shape = (200, 200)
params.preproc.tools.equalize_hist_adapt.nbins = 65535

params.preproc.tools.equalize_hist_local.enable = False
params.preproc.tools.equalize_hist_local.radius = 10

params.preproc.tools.rescale_intensity.enable = False
params.preproc.tools.rescale_intensity.minima = 0.
params.preproc.tools.rescale_intensity.maxima = 65535.

topology = TopologyPreproc(params)
topology.compute(sequential=False)
