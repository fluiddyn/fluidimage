from fluidimage.experimental.topologies.image2image_new import TopologyImage2Image
from fluidimage.experimental.executors.executor_await import ExecutorAwaitMultiprocs

params = TopologyImage2Image.create_default_params()

params.series.path = '../../../image_samples/Karman/Images2'
params.series.ind_start = 1
params.series.ind_step = 1

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'recompute'
params.saving.postfix = 'await_piv2_recompute'

params.im2im = ('1','2')


topology = TopologyImage2Image(params, logging_level='info')
# topology.make_code_graphviz('topo.dot')

executer = ExecutorAwaitMultiprocs(topology, multi_executor=True, sleep_time=0.1,
                                   worker_limit=4, queues_limit=5)
executer.compute()
