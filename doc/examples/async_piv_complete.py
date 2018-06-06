from fluidimage.topologies.piv_async import Async_piv
from fluidimage.topologies.base_async import Base_async
from fluidimage.topologies.piv import TopologyPIV
from fluidimage.works.piv import multipass

params = TopologyPIV.create_default_params()
# sub_path_image = "Images2"
# path = "../../image_samples/Karman/{}/".format(sub_path_image)
# path_save = "../../image_samples/Karman/{}.results.async/".format(sub_path_image)
#

params.series.path = '../../image_samples/Karman/'
params._set_child(
    "path", attribs={"sub_images_path": "Images2"}
)

params.series.ind_start = 1
params.series.ind_stop = 20
params.series.ind_step = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'complete'
params.saving.postfix = 'async_piv_complete'


work = multipass.WorkPIV(params)
async_proc_class = Async_piv
topology = Base_async(params, work, async_proc_class, logging_level='info')
topology.compute()
