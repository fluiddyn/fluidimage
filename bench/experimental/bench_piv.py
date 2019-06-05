
from os import symlink
from pathlib import Path
from shutil import rmtree
from time import time

from fluidimage import path_image_samples
from fluidimage.experimental.topologies.piv_new import TopologyPIV
from fluidimage.topologies.piv import TopologyPIV as OldTopologyPIV

path_dir_images = path_image_samples / "Milestone/Images"
path_images = list(path_dir_images.glob("im*"))
path_images.sort()

nb_couples = 12

names = []
for ind in range(nb_couples):
    names.append(f"im{ind:04d}a.png")
    names.append(f"im{ind:04d}b.png")

path_dir_many = path_dir_images.with_name("Images_many")

paths_images_link = list(path_dir_many.glob("im*"))

if len(paths_images_link) != 2*nb_couples:

    if path_dir_many.exists():
        rmtree(path_dir_many)

    path_dir_many.mkdir(exist_ok=True)

    for ind in range(nb_couples):
        symlink(path_images[0], path_dir_many / f"im{ind:04d}a.png")
        symlink(path_images[1], path_dir_many / f"im{ind:04d}b.png")

path_src = path_dir_many / "im*"
postfix = "bench_piv_new"

def modify_params(params):
    params.series.path = str(path_src)
    params.series.ind_start = 0
    params.series.strcouple = "i, 0:2"

    params.piv0.shape_crop_im0 = 192
    params.piv0.method_correl = 'fftw'
    params.piv0.displacement_max = "50%"

    params.mask.strcrop = ':, :'

    params.multipass.number = 2
    params.multipass.use_tps = 'last'
    params.multipass.smoothing_coef = 10.
    params.multipass.threshold_tps = 0.1

    params.fix.correl_min = 0.07
    params.fix.threshold_diff_neighbour = 10

    params.saving.how = "recompute"
    params.saving.postfix = postfix


def bench():
    params = TopologyPIV.create_default_params()
    modify_params(params)

    """

There is a big problem with the exec_async executor. I guess that the PIV work is
not thread safe (maybe the FFT used for the correlation).

    """

    executors = [
        "exec_sequential",
        "multi_exec_async",
        "exec_async_multi",
        "exec_async_servers",
    ]

    durations = []

    for executor in executors:
        t_start = time()
        topology = TopologyPIV(params, logging_level="info")
        topology.compute(executor, sleep_time=0.01, stop_if_error=False)
        durations.append(time() - t_start)

    t_start = time()
    topology = OldTopologyPIV(params, logging_level="info")
    topology.compute()
    duration_old = time() - t_start
    durations.append(duration_old)
    executors.append("old")

    duration_norm = durations[0]

    for executor, duration in zip(executors, durations):
        print(
            f"{executor + ':':30s}{duration:10.2f} s, "
            f"speedup = {duration_norm/duration}"
        )

    path_out = Path(str(path_src.parent) + "." + postfix)
    if path_out.exists():
        rmtree(path_out)

if __name__ == "__main__":
    bench()
