from fluidimage.topologies.surface_tracking import TopologySurfaceTracking

params = TopologySurfaceTracking.create_default_params()

params.film.path = '../../../surfacetracking/111713'
params.film.pathRef = '../../../surfacetracking/reference_water'

# params.saving.how has to be equal to 'complete' for idempotent jobs
# (on clusters)
params.saving.how = 'complete'
params.saving.postfix = 'surface_tracking_complete'

topology = TopologySurfaceTracking(params, logging_level='info')
#topology.make_code_graphviz('topo.dot')
topology.compute(None)
