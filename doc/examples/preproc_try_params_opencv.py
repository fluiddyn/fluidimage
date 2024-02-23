from fluidimage.preproc import Work

params = Work.create_default_params(backend="opencv")

params.preproc.series.path = "../../image_samples/Karman/Images"
print("Available preprocessing tools: ", params.preproc.tools.available_tools)

params.preproc.tools.sliding_median.enable = True
params.preproc.tools.sliding_median.window_size = 25

preproc = Work(params)

preproc.display(1, hist=False)
