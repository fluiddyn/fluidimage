import sys

import numpy as np
import pytest

from fluiddyn.io.image import imsave
from fluidimage.executors import supported_multi_executors
from fluidimage.topologies.mean import Topology, main

executors = [
    "exec_sequential",
    "exec_async_sequential",
    "exec_async",
    "exec_async_multi",
    "exec_async_servers",
    "exec_async_servers_threading",
]

executors.extend(supported_multi_executors)

num_images = 19


@pytest.fixture(scope="session")
def tmp_path_images(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dir_images")
    path_dir_images = tmp_path / "images"
    path_dir_images.mkdir()

    im = np.empty([4, 8], dtype=np.uint8)

    for idx in range(num_images):
        im.fill(idx)
        name = f"im{idx:03d}.png"
        imsave(path_dir_images / name, im, as_int=True)

    return path_dir_images


@pytest.mark.parametrize("executor", executors)
def test_mean(tmp_path_images, executor):

    mean_should_be = num_images // 2

    tmp_path = tmp_path_images

    params = Topology.create_default_params()
    params.images.path = str(tmp_path)
    params.saving.postfix = executor

    topology = Topology(params)

    topology.compute(executor, nb_max_workers=2)

    result = topology.result

    assert result[0, 0] == mean_should_be, (
        result[0, 0],
        mean_should_be,
    )
    assert np.all(np.isclose(result, result[0, 0]))

    assert len(topology.results) == num_images
    assert topology.path_result.exists()


def test_mean_help(monkeypatch):

    command = "fluidimage-mean -h"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())


def test_mean_command(tmp_path_images, monkeypatch):

    tmp_path = tmp_path_images
    path_output = tmp_path / "a-great-name.png"

    command = f"fluidimage-monitor {tmp_path} -o {path_output.with_suffix('')}"
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()

    assert path_output.exists()
