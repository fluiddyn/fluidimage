from fluidimage.image2image import Work

params = Work.create_default_params()

# for a function:
# params.im2im = 'fluidimage.image2image.im2im_func_example'

# for a class (with one argument for the function init):
params.im2im = "fluidimage.image2image.Im2ImExample"
params.args_init = ((1024, 2048), "clip")

params.images.path = "../../image_samples/Jet/Images/c*"
params.images.str_subset = "60:,:"

work = Work(params)

work.display()
