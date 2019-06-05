from fluidimage.preproc.base import PreprocBase

params = PreprocBase.create_default_params(backend="opencv")

params.preproc.series.path = "../../image_samples/Karman/Images"
print("Available preprocessing tools: ", params.preproc.tools.available_tools)

params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 25

preproc = PreprocBase(params)
preproc()

preproc.display(1, hist=False)
