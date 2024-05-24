from pathlib import Path
from shutil import rmtree

from pyinstrument import Profiler
from pyinstrument.renderers import ConsoleRenderer

from fluidimage.topologies.mean import Topology

path_images = Path("/fsnet/project/watu/2022/22INTERNAL_F/DATA/EXP44/PCO_50mm")
path_images = Path("/data/PCO_50mm")
rmtree(path_images.parent / "PCO_50mm.mean", ignore_errors=True)

params = Topology.create_default_params()
params.images.path = str(path_images / "im*.png")
params.images.str_subset = ":100"

topology = Topology(params)
executor = "exec_sequential"
executor = "exec_async_sequential"
# executor = "multi_exec_sync"
executor = "multi_exec_async"

profiler = Profiler()
profiler.start()
topology.compute(executor=executor, nb_max_workers=4)
profiler.stop()

print(
    profiler.output(
        renderer=ConsoleRenderer(
            unicode=True,
            color=True,
            show_all=False,
            # time="percent_of_total",
            # flat=True,  # buggy with pyinstrument 4.6.2!
        )
    )
)
