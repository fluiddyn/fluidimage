from fluidimage.experimental.executors.executor_await import (
    ExecutorAwaitMultiprocs
)
from fluidimage.experimental.topologies.example import TopologyExample

path_dir = "../../../image_samples/Karman/Images3"
topology = TopologyExample(logging_level='info', path_output="../../../image_samples/Karman/Images2.example_cpu_bounded", path_dir=path_dir)
#topology.make_code_graphviz('topo.dot')


topology.compute(executer=ExecutorAwaitMultiprocs(topology, multi_executor=False), sequential=False)
