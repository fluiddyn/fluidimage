from fluidimage.preproc import Work

params = Work.create_default_params()

params.preproc.series.path = "../../image_samples/Karman/Images"
print("Available preprocessing tools: ", params.preproc.tools.available_tools)

params.preproc.tools.sequence = ["sliding_median", "global_threshold"]
params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 25

params.preproc.tools.global_threshold.enable = True
params.preproc.tools.global_threshold.minima = 0.0
params.preproc.tools.global_threshold.maxima = 255.0

preproc = Work(params)

preproc.display(1, hist=False)
