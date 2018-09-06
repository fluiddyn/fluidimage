
from shutil import rmtree
from pathlib import Path
from time import time

from fluidimage.experimental.topologies.piv_new import TopologyPIV

from fluidimage import path_image_samples

path_Jet = path_image_samples / "Jet/Images/c06*"
postfix = "bench_piv_new"


def bench():
    params = TopologyPIV.create_default_params()

    params.series.path = str(path_Jet)
    params.series.ind_start = 60
    params.series.ind_step = 1
    params.series.strcouple = "i, 0:2"

    params.piv0.shape_crop_im0 = 128
    params.multipass.number = 3
    params.multipass.use_tps = True

    # params.saving.how has to be equal to 'complete' for idempotent jobs
    # (on clusters)
    params.saving.how = "recompute"
    params.saving.postfix = postfix

    executors = [
        "exec_async",
        "multi_exec_async",
        "exec_async_multi",
        "exec_async_servers",
    ]

    durations = []

    for executor in executors:
        t_start = time()
        topology = TopologyPIV(params, logging_level="info")
        topology.compute(executor)
        durations.append(time() - t_start)

    for executor, duration in zip(executors, durations):
        print(f"{executor + ':':30s}{duration}")

    path_out = Path(str(path_Jet.parent) + "." + postfix)
    if path_out.exists():
        rmtree(path_out)

if __name__ == "__main__":
    bench()
