"""Testing utilities"""

import shutil

from pytest import fixture

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
