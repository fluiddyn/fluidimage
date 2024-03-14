"""Testing utilities"""

import shutil

import matplotlib.pyplot as plt
from pytest import fixture

from fluiddyn.io.image import imread, imsave
from fluidimage import get_path_image_samples

path_image_samples = get_path_image_samples()


def _create_dir(tmp_path, name):
    path_dir_images = tmp_path / "Images"
    path_dir_images.mkdir()
    for path_im in (path_image_samples / name / "Images").glob("*"):
        shutil.copy(path_im, path_dir_images)
    return path_dir_images


@fixture
def tmp_path_jet(tmp_path):
    return _create_dir(tmp_path, "Jet")


@fixture
def tmp_path_karman(tmp_path):
    return _create_dir(tmp_path, "Karman")


@fixture
def tmp_path_oseen(tmp_path):
    return _create_dir(tmp_path, "Oseen")


def _create_dir_small(tmp_path_factory, dataset: str):
    path_in = get_path_image_samples() / dataset / "Images"
    tmp_path = tmp_path_factory.mktemp("dir_" + dataset)
    for path in sorted(path_in.glob("*")):
        name = path.name
        im = imread(path)
        im = im[::6, ::6]
        imsave(tmp_path / name, im, as_int=True)
    return tmp_path


@fixture(scope="session")
def tmp_path_karman_small(tmp_path_factory):
    return _create_dir_small(tmp_path_factory, "Karman")


@fixture(scope="session")
def tmp_path_jet_small(tmp_path_factory):
    return _create_dir_small(tmp_path_factory, "Jet")


@fixture
def close_plt_figs():
    plt.close("all")
    yield
    plt.close("all")
