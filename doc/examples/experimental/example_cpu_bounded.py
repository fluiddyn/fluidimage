from fluidimage.topologies.experimental.topology_example import Topology_example
from fluidimage.topologies.experimental.executer_await_multiproc_tcp import ExecuterAwaitMultiprocs
from fluidimage.topologies.experimental.executer_await import ExecuterAwaitMultiprocs
import os

path_dir = "../../../image_samples/Karman/Images3"
topology = Topology_example(logging_level='info', path_output="../../../image_samples/Karman/Images2.example_cpu_bounded", path_dir=path_dir)
#topology.make_code_graphviz('topo.dot')


topology.compute(executer=ExecuterAwaitMultiprocs(topology, multi_executor=False), sequential=False)
