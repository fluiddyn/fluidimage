import os
from pathlib import Path

path_base = os.getenv("DIR_DATA_PIV_CHALLENGE")
if path_base is None:
    possible_paths = []
    with open("possible_paths.txt", encoding="utf-8") as file:
        for line in file:
            possible_paths.append(os.path.expanduser(line.strip()))

    ok = False
    for path_base in possible_paths:
        if os.path.exists(path_base):
            ok = True
            break

    if not ok:
        raise ValueError(
            "One of the base paths has to be created. "
            "You can also add possible paths in the file possible_paths.txt."
        )

paths = Path(path_base).glob("PIV*")
paths = {path.name[3:]: path for path in paths}


def get_path(key):
    return paths[key] / "Images"
