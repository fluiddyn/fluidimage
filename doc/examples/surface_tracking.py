from fluidimage.topologies.surface_tracking import TopologySurfaceTracking

params = TopologySurfaceTracking.create_default_params()

params.film.fileName = 'film.cine'
params.film.path = '../../../surfacetracking/111713'
params.film.path_ref = '../../../surfacetracking/reference_water'

# params.saving.how has to be equal to 'complete' for idempotent jobs
# # (on clusters)
params.saving.plot = False
params.saving.how_many = 100
params.saving.how = 'complete'
params.saving.postfix = 'surface_tracking_complete'

topology = TopologySurfaceTracking(params, logging_level='info')
#topology.make_code_graphviz('topo.dot')
seq = False
topology.compute(sequential=seq)

# not generating plots if seq mode is false
if seq == False:
    params.saving.plot = False
