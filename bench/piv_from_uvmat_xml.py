
import sys
from time import time
import socket

from fluidimage.topologies.piv import TopologyPIV

from fluidimage import config_logging

from fluidimage.run_from_xml import params_from_uvmat_xml, InstructionsUVMAT

base = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/Bicouche'

nb_cores = int(sys.argv[-1])

host = socket.gethostname()

config_logging('info')

instructions = InstructionsUVMAT(
    path_file=base + '/Dalsa1.civ_bench_cluster_xml/0_XML/img_1-161.xml')

params = params_from_uvmat_xml(instructions)

params.saving.path = base + '/Dalsa1.piv_bench_' + host + '_{}cores'.format(
    nb_cores)
params.saving.how = 'complete'

params.multipass.subdom_size = 500

topology = TopologyPIV(params)

t = time()
topology.compute()
t = time() - t
print('ellapsed time: {}s'.format(t))
