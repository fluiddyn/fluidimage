import subprocess
import sys

import pytest

from fluidimage.executors import supported_multi_executors
from fluidimage.gui.monitor import MonitorApp, main
from fluidimage.piv import TopologyPIV

postfix = "test_monitor"


def test_monitor_help(monkeypatch):

    command = "fluidimage-monitor -h"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        try:
            main()
        except SystemExit:
            pass
        else:
            raise RuntimeError


def test_monitor_version(monkeypatch):
    command = "fluidimage-monitor --version"

    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()


def test_monitor_bad_paths(monkeypatch, tmp_path):

    command = "fluidimage-monitor __dir_does_not_exist__"
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()

    command = f"fluidimage-monitor {tmp_path}"
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "argv", command.split())
        main()


@pytest.mark.usefixtures("close_plt_figs")
@pytest.mark.parametrize("executor", supported_multi_executors)
@pytest.mark.asyncio
async def test_monitor(monkeypatch, tmp_path_oseen, executor):
    path_dir_images = tmp_path_oseen

    params = TopologyPIV.create_default_params()

    params.series.path = str(path_dir_images)

    params.piv0.shape_crop_im0 = 64
    params.multipass.number = 1
    params.multipass.use_tps = False

    params.saving.how = "recompute"
    params.saving.postfix = postfix

    topology = TopologyPIV(params, logging_level="info")
    topology.compute(executor, nb_max_workers=2)

    with monkeypatch.context() as ctx:
        ctx.setattr(
            sys, "argv", ["fluidimage-monitor", str(topology.path_dir_result)]
        )
        args = MonitorApp.parse_args()

    app = MonitorApp(args)

    async with app.run_test() as pilot:
        await pilot.press("p")
        await pilot.press("i")

        def _run(*args, **kwargs):
            pass

        with monkeypatch.context() as ctx:
            ctx.setattr(subprocess, "run", _run)
            await pilot.press("f")

    app.update_info()

    node_saving = app.tree_params.root.children[0]

    class MyEvent:
        def __init__(self, node):
            self.node = node

    event = MyEvent(node_saving)
    app.on_tree_node_selected(event)

    leaf = node_saving.children[0]
    event.node = leaf
    app.on_tree_node_selected(event)
