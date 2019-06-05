
import os
import socket
import sys
from glob import glob

from fluidimage.run_from_xml import (
    ParamContainer,
    params_from_uvmat_xml,
    tidy_uvmat_instructions,
)
from fluidimage.topologies.piv import TopologyPIV

base = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/Bicouche'

host = socket.gethostname()

base += '/Dalsa1.civ_bench_' + host + '_xml/0_XML/img_*.xml'

print(base)

xml_file = glob(base)[0]

instructions = ParamContainer(path_file=xml_file)
instructions.Action.ActionName
tidy_uvmat_instructions(instructions)

params = params_from_uvmat_xml(instructions)

params.series.ind_start = 0

params.saving.path = os.path.split(
    os.path.dirname(xml_file))[0].replace('.civ', '.piv')

try:
    nb_cores = int(sys.argv[-1])
except ValueError:
    nb_cores = None
else:
    params.saving.path += f'_{nb_cores}cores'


params.saving.how = 'recompute'

# params.multipass.subdom_size = 100

print('params.multipass.subdom_size = ', params.multipass.subdom_size)

params.multipass.use_tps = True

topology = TopologyPIV(params, logging_level='info', nb_max_workers=None)

topology.compute()
