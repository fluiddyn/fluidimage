import os
from fluidimage.experimental.no_topology_computations.piv_async import AsyncPIV
from fluidimage.experimental.no_topology_computations.base_async import BaseAsync
from fluidimage.topologies.surface_tracking import TopologySurfaceTracking
from fluidimage.works.surfaceTracking.surface_tracking import WorkSurfaceTracking

params = TopologySurfaceTracking.create_default_params()

params.film.fileName = 'film.cine'
params.film.path = '../../../surfacetracking/111713'
params.film.pathRef = '../../../surfacetracking/reference_water'
#
params.series.path = '../../../surfacetracking/111713'
# params._set_child(
#     "path", attribs={"sub_images_path": ""}
# )

params.series.ind_start = 1
params.series.ind_stop = len(os.listdir(params.series.path +params.path.sub_images_path)) - 1
params.series.ind_step = 1



# params.saving.how has to be equal to 'complete' for idempotent jobs
# # (on clusters)
params.saving.plot = False
params.saving.how_many = 100
params.saving.how = 'complete'
params.saving.postfix = 'surface_tracking_complete'

topology = TopologySurfaceTracking(params, logging_level='info')
#topology.make_code_graphviz('topo.dot')
seq = True
async_proc_class = AsyncPIV
work = WorkSurfaceTracking(params)
topology = BaseAsync(params, work, async_proc_class, logging_level='info')
topology.compute()

#Force not generating plots if seq mode is false
if seq == False:
    params.saving.plot = False