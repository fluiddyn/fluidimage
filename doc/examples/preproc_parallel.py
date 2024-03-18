from fluidimage import get_path_image_samples
from fluidimage.preproc import Topology

params = Topology.create_default_params()

params.series.path = get_path_image_samples() / "Jet/Images"
params.series.str_subset = "i,:"

p_tools = params.tools

print("Available preprocessing tools: ", p_tools.available_tools)
p_tools.sequence = ["temporal_median", "sliding_median", "global_threshold"]
print("Enabled preprocessing tools: ", p_tools.sequence)

p_tools.sliding_median.enable = True
p_tools.sliding_median.weight = 0.5
p_tools.sliding_median.window_size = 10

p_tools.temporal_median.enable = False
p_tools.temporal_median.weight = 0.5
p_tools.temporal_median.window_shape = (5, 2, 2)

p_tools.global_threshold.enable = True
p_tools.global_threshold.minima = 0.0

params.saving.how = "recompute"
params.saving.postfix = "pre_example"

topology = Topology(params, logging_level="info", nb_max_workers=4)

# Compute in parallel
topology.compute()

# Compute in sequential (for debugging)
# topology.compute(sequential=True)

assert len(topology.results) == 2
