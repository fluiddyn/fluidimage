

import os
from glob import glob


path_base = os.getenv('DIR_DATA_PIV_CHALLENGE')
if path_base is None:
    path_base = '/fsnet/project/meige/2016/16FLUIDIMAGE/samples/pivchallenge'

paths = glob(os.path.join(path_base, 'PIV*'))

paths = {os.path.split(path)[-1][3:]: path for path in paths}


def get_path(key):
    return os.path.join(paths[key], 'Images')
