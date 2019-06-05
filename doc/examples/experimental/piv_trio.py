import os

from fluidimage.experimental.no_topology_computations.base_trio import BaseAsync
from fluidimage.experimental.no_topology_computations.piv_trio import PivTrio
from fluidimage.topologies.piv import TopologyPIV
from fluidimage.works.piv import multipass

params = TopologyPIV.create_default_params()

params.series.path = '../../../image_samples/Karman/Images3'


params.series.ind_start = 1
params.series.ind_stop = len(os.listdir(params.series.path )) - 1
params.series.ind_step = 1

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

# params.saving.how has to be equal to 'complete' for idempotent jobs
params.saving.how = 'complete'
params.saving.postfix = 'async_piv_complete'

#Create work, async processus and topologie
work = multipass.WorkPIV(params)
async_proc_class = PivTrio
topology = BaseAsync(params, work, async_proc_class, logging_level='info')
topology.compute()
