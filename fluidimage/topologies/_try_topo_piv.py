
from fluidimage.topologies.piv import TopologyPIV

from fluidimage import config_logging

config_logging("info")


params = TopologyPIV.create_default_params()

# path = '../../image_samples/Oseen/Images/Oseen_center*'
path = "../../image_samples/Karman/Images"

# path = '../../image_samples/Jet/Images/c*'
# params.series.strcouple = 'i+60, 0:2'
# params.series.strcouple = 'i+60:i+62, 0'

params.series.path = path

params.piv0.shape_crop_im0 = 32
params.multipass.number = 2
params.multipass.use_tps = False

topology = TopologyPIV(params)

# topology.compute(sequential=False)

# topology.make_code_graphviz('topo.dot')
