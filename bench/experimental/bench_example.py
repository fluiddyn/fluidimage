
from pathlib import Path
from shutil import rmtree
from time import time

from fluidimage import path_image_samples
from fluidimage.experimental.topologies.example import TopologyExample

path_input = path_image_samples / "Karman/Images"
path_dir_result = path_input.parent / f"Images.bench"

def bench():
    params = TopologyExample.create_default_params()

    params["path_input"] = path_input
    params["path_dir_result"] = path_dir_result
    params["nloops"] = 30000
    params["multiplicator_nb_images"] = 2

    executors = [
        "exec_sequential",
        "exec_async_sequential",
        "exec_async",
        "multi_exec_async",
        "exec_async_multi",
        "exec_async_servers",
    ]

    durations = []

    for executor in executors:
        t_start = time()
        topology = TopologyExample(params, logging_level="info")
        topology.compute(executor, sleep_time=0.01, nb_max_workers=None)
        durations.append(time() - t_start)

    duration_seq = durations[0]

    for executor, duration in zip(executors, durations):
        print(
            f"{executor + ':':30s}{duration:8.2f} s, "
            f"speedup: {duration_seq/duration:5.2f}"
        )

    # if path_dir_result.exists():
    #     rmtree(path_dir_result)

if __name__ == "__main__":
    bench()
