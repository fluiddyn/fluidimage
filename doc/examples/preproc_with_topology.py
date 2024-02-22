from fluidimage import get_path_image_samples
from fluidimage.topologies.preproc import TopologyPreproc

params = TopologyPreproc.create_default_params()

params.preproc.series.path = get_path_image_samples() / "Jet/Images"
params.preproc.series.str_subset = "i+60:i+62, 0"

print("Available preprocessing tools: ", params.preproc.tools.available_tools)
params.preproc.tools.sequence = [
    "temporal_median",
    "sliding_median",
    "global_threshold",
]
print("Enabled preprocessing tools: ", params.preproc.tools.sequence)

params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.weight = 0.5
params.preproc.tools.sliding_median.window_size = 10

params.preproc.tools.temporal_median.enable = False
params.preproc.tools.temporal_median.weight = 0.5
params.preproc.tools.temporal_median.window_shape = (5, 2, 2)

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 0.0

topology = TopologyPreproc(params, logging_level="info")

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)
