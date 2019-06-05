from fluidimage.experimental.executors.executor_await import (
    ExecutorAwaitMultiprocs
)
from fluidimage.experimental.topologies.image2image_new import TopologyImage2Image
from fluidimage.preproc import image2image

params = TopologyImage2Image.create_default_params()

params.series.path = '../../../image_samples/Karman/Images2'
params.series.ind_start = 1
params.series.ind_step = 1

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'recompute'
params.saving.postfix = 'await_im2im_recompute'


params.im2im = 'fluidimage.preproc.image2image.im2im_func_example'
topology = TopologyImage2Image(params, logging_level='info')
image2image.complete_im2im_params_with_default(params)

topology.make_code_graphviz('topo.dot')

executer = ExecutorAwaitMultiprocs(topology, multi_executor=True, sleep_time=0.1,
                                   worker_limit=4, queues_limit=5)
executer.compute()
