from fluidimage.experimental.executors.executor_classic import ExecutorClassic
from fluidimage.experimental.topologies.piv_new import TopologyPIV

params = TopologyPIV.create_default_params()

params.series.path = '../../../image_samples/Karman/Images2'
params.series.ind_start = 1
params.series.ind_step = 1


params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = True

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'recompute'
params.saving.postfix = 'classic_piv2_recompute'


topology = TopologyPIV(params, logging_level='info')
# topology.make_code_graphviz('topo.dot')

executer = ExecutorClassic(topology)
executer.compute()
