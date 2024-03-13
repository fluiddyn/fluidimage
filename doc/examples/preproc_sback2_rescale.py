from fluidimage.preproc import Topology

params = Topology.create_default_params()

params.series.path = (
    "$HOME/useful/project/16MILESTONE/Data/"
    + "Exp21_2016-06-22_N0.8_L6.0_V0.08_piv3d/PCO_top/level2.tfilter"
)

params.series.str_subset = "i:i+1"
params.series.ind_start = 1200
params.series.ind_stop = 1224

params.saving.postfix = "rescale"
params.saving.format = "img"

p_tools = params.tools

p_tools.sequence = [
    "equalize_hist_global",
    "equalize_hist_adapt",
    "equalize_hist_local",
    "rescale_intensity",
]

p_tools.equalize_hist_global.enable = False
p_tools.equalize_hist_global.nbins = 500

p_tools.equalize_hist_adapt.enable = True
p_tools.equalize_hist_adapt.window_shape = (200, 200)
p_tools.equalize_hist_adapt.nbins = 65535

p_tools.equalize_hist_local.enable = False
p_tools.equalize_hist_local.radius = 10

p_tools.rescale_intensity.enable = False
p_tools.rescale_intensity.minima = 0.0
p_tools.rescale_intensity.maxima = 65535.0

topology = Topology(params)
topology.compute(sequential=False)
