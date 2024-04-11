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
        await pilot.press("r")
