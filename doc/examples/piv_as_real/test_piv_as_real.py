import shutil
import sys
from pathlib import Path
from runpy import run_path

from .params_piv import get_path

path_here = Path(__file__).absolute().parent
sys.path.insert(0, str(path_here))


def test_scripts(monkeypatch):

    # first clean up
    path_images_dir = get_path(0)

    for suffix in (".pre", ".pre.piv"):
        path = path_images_dir.with_suffix(suffix)
        shutil.rmtree(path, ignore_errors=True)

    for post in ("pre", "piv"):

        run_path(path_here / f"try_{post}.py")

        name = f"params_{post}.py"
        with monkeypatch.context() as ctx:
            ctx.setattr(sys, "argv", [name, "0"])
            run_path(path_here / name, run_name="__main__")

        name = f"job_{post}.py"
        with monkeypatch.context() as ctx:
            ctx.setattr(sys, "argv", [name, "0", "--nb-cores", "2"])
            run_path(path_here / name, run_name="__main__")
