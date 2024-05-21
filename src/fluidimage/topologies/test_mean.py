import numpy as np
import pytest

from fluiddyn.io.image import imsave
from fluidimage.topologies.mean import TopologyMeanImage as Topology


@pytest.fixture(scope="session")
def tmp_path_19_images(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dir_19_images")
    path_dir_images = tmp_path / "Images"
    path_dir_images.mkdir()

    im = np.empty([4, 8], dtype=np.uint8)

    for idx in range(19):
        im.fill(idx)
        name = f"im{idx:03d}.png"
        imsave(path_dir_images / name, im, as_int=True)

    return path_dir_images


def test_mean(tmp_path_19_images):

    tmp_path = tmp_path_19_images

    params = Topology.create_default_params()
    params.images.path = str(tmp_path)

    topology = Topology(params)

    executor = "exec_sequential"
    topology.compute(executor, nb_max_workers=2)
