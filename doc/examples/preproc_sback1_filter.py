from fluidimage.preproc import Topology

params = Topology.create_default_params()

params.preproc.series.path = (
    "$HOME/useful/project/16MILESTONE/Data/"
    "Exp21_2016-06-22_N0.8_L6.0_V0.08_piv3d/PCO_top/level2"
)

params.preproc.series.str_subset = "i:i+23"
params.preproc.series.ind_start = 1200
params.preproc.series.ind_stop = 1202

params.preproc.saving.postfix = "temporal"
params.preproc.saving.format = "img"

params.preproc.tools.sequence = [
    "temporal_median",
    "temporal_percentile",
    "temporal_minima",
]

params.preproc.tools.temporal_median.enable = False
params.preproc.tools.temporal_median.weight = 1.0

params.preproc.tools.temporal_percentile.enable = True
params.preproc.tools.temporal_percentile.percentile = 10.0

params.preproc.tools.temporal_minima.enable = False
params.preproc.tools.temporal_minima.weight = 1.0

topology = Topology(params)
topology.compute(sequential=False)
