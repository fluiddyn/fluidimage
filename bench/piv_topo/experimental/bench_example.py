
from shutil import rmtree
from pathlib import Path
from time import time

from fluidimage.experimental.topologies.example import TopologyExample

from fluidimage import path_image_samples

path_input = path_image_samples / "Karman/Images"
path_dir_result = path_input.parent / f"Images.bench"

def bench():
    params = TopologyExample.create_default_params()

    params["path_input"] = path_input
    params["path_dir_result"] = path_dir_result
    params["nloops"] = 10000

    executors = [
        "exec_async",
        "exec_async_sequential",
        # "multi_exec_async",
        # "exec_async_multi",
        # "exec_async_servers",
    ]

    durations = []

    for executor in executors:
        t_start = time()
        topology = TopologyExample(params, logging_level="info")
        topology.compute(executor)
        durations.append(time() - t_start)

    for executor, duration in zip(executors, durations):
        print(f"{executor + ':':30s}{duration}")

    # if path_dir_result.exists():
    #     rmtree(path_dir_result)

if __name__ == "__main__":
    bench()
