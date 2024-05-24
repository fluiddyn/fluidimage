from shutil import rmtree

from path_images import get_path
from pyinstrument import Profiler
from pyinstrument.renderers import ConsoleRenderer

from fluidimage.topologies.mean import Topology

path_images = get_path("2005C")
rmtree(path_images.parent / "Images.mean", ignore_errors=True)

params = Topology.create_default_params()
params.images.path = str(path_images / "c*.bmp")

topology = Topology(params)
executor = "exec_sequential"
executor = "exec_async_sequential"

profiler = Profiler()
profiler.start()
topology.compute(executor=executor, nb_max_workers=2)
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
