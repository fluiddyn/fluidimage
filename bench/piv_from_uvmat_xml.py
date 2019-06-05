
import socket
import sys

from fluidimage.run_from_xml import ParamContainer, params_from_uvmat_xml
from fluidimage.topologies.piv import TopologyPIV

base = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/Bicouche'

host = socket.gethostname()

instructions = ParamContainer(
    path_file=base + '/Dalsa1.civ_bench_cluster_xml/0_XML/img_1-161.xml')

params = params_from_uvmat_xml(instructions)

params.saving.path = base + '/Dalsa1.piv_bench_' + host

try:
    nb_cores = int(sys.argv[-1])
except ValueError:
    nb_cores = None
else:
    params.saving.path += f'_{nb_cores}cores'


params.saving.how = 'complete'

params.multipass.subdom_size = 500

topology = TopologyPIV(params)

# topology.compute()
