
from fluidimage.topologies.piv import TopologyPIV

params = TopologyPIV.create_default_params()

# path = '../../image_samples/Oseen/Images/Oseen_center*'
path = '../../image_samples/Karman/Images'

# path = '../../image_samples/Jet/Images/c*'
# params.series.strcouple = 'i+60, 0:2'
# params.series.strcouple = 'i+60:i+62, 0'

params.series.path = path

topology = TopologyPIV(params)

topology.compute(sequential=False)

# topology.make_code_graphviz('topo.dot')
# then the graph can be produced with the command:
# dot topo.dot -Tpng -o topo.png
# dot topo.dot -Tx11
