

import os
from glob import glob

path_base = os.getenv('DIR_DATA_PIV_CHALLENGE')
if path_base is None:
    possible_paths = []
    with open('possible_paths.txt') as f:
        for line in f:
            possible_paths.append(os.path.expanduser(line.strip()))

    ok = False
    for path_base in possible_paths:
        if os.path.exists(path_base):
            ok = True
            break

    if not ok:
        raise ValueError(
            'One of the base paths has to be created. '
            'You can also add possible paths in the file possible_paths.txt.')

paths = glob(os.path.join(path_base, 'PIV*'))

paths = {os.path.split(path)[-1][3:]: path for path in paths}


def get_path(key):
    return os.path.join(paths[key], 'Images')
