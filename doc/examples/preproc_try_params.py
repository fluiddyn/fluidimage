from fluidimage.preproc import Work

params = Work.create_default_params()

params.series.path = "../../image_samples/Karman/Images"
print("Available preprocessing tools: ", params.tools.available_tools)

params.tools.sequence = ["sliding_median", "global_threshold"]
params.tools.sliding_median.enable = True
params.tools.sliding_median.window_size = 25

params.tools.global_threshold.enable = True
params.tools.global_threshold.minima = 0.0
params.tools.global_threshold.maxima = 255.0

preproc = Work(params)

preproc.display(1, hist=False)
