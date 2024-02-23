from fluidimage import get_path_image_samples
from fluidimage.works.bos import WorkBOS


def test_work_bos():
    path_input_files = get_path_image_samples() / "Karman/Images"

    params = WorkBOS.create_default_params()

    params.images.path = str(path_input_files)
    params.images.str_subset = "1:3"

    params.piv0.shape_crop_im0 = 32
    params.multipass.number = 2
    params.multipass.use_tps = False

    params.mask.strcrop = ":, 50:500"

    # temporary, avoid a bug on Windows
    params.piv0.method_correl = "pythran"
    params.piv0.shape_crop_im0 = 16

    # compute only few vectors
    params.piv0.grid.overlap = -8

    work = WorkBOS(params)

    assert not hasattr(work, "process_1_serie")

    result = work.process_1_image()

    print(type(result))
