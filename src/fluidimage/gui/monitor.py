"""Monitor computations done with topologies

.. autoclass:: MonitorApp
   :members:
   :private-members:


"""

import argparse
import os
import subprocess
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    Rule,
    TabbedContent,
    TabPane,
    Tree,
)

from fluidimage import ParamContainer


def build_branch(branch, params_node):
    print(params_node)
    branch.root.expand()
    for key in params_node._get_key_attribs():
        branch.root.add_leaf(f"{key} = {params_node[key]}")


class MonitorApp(App):
    """Fluidimage monitor Textual app"""

    TITLE = "Fluidimage monitor app"
    SUB_TITLE = "Monitoring parallel Fluidimage computations"

    BINDINGS = [
        Binding(key="q", action="quit", description="Quit the app"),
        Binding(
            key="i",
            action="show_info",
            description="Show info",
        ),
        Binding(
            key="p",
            action="show_params",
            description="Show parameters",
        ),
        Binding(
            key="f",
            action="launch_fluidpivviewer",
            description="Launch fluidpivviewer",
        ),
    ]

    @classmethod
    def parse_args(cls):

        parser = argparse.ArgumentParser(
            description=cls.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "path",
            help="Path file or directory.",
            type=str,
            nargs="?",
            default=os.getcwd(),
        )
        parser.add_argument(
            "-v", "--verbose", help="verbose mode", action="count"
        )

        return parser.parse_args()

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.path_in = Path(self.args.path)

        if not self.path_in.exists():
            print(f"{self.args.path} does not exist.")
            self.exit(0)

        try:
            self.path_job_info = sorted(self.path_in.glob("job_*"))[-1]
        except IndexError:
            print("No job info folder found.")
            self.exit(0)

        self.path_lockfile = self.path_job_info / "is_running.lock"
        self.job_is_running = self.path_lockfile.exists()

        self.path_info = self.path_job_info / "info.xml"
        info = ParamContainer(path_file=self.path_info)
        self.info_job = {
            key: info[key] for key in sorted(info._get_key_attribs())
        }

        self.params = ParamContainer(path_file=self.path_job_info / "params.xml")

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Info", id="info"):

                with Middle():
                    with Center():
                        yield Label(f"Output directory: {str(self.path_in)}")
                        yield Label(f"Running: {self.job_is_running}")
                    yield Rule()
                    yield DataTable()
                    yield Rule()
                    with Center():
                        yield ProgressBar()

            with TabPane("Parameters", id="params"):
                with Middle():
                    with Center():
                        yield Label("Parameters")
                        self.tree_params = Tree("params")
                        yield self.tree_params

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Parameter", "Value")
        table.add_rows([(key, value) for key, value in self.info_job.items()])

        build_branch(self.tree_params, self.params)

    def action_launch_fluidpivviewer(self) -> None:
        print("launching fluidpivviewer")
        subprocess.run(["fluidpivviewer", str(self.path_in)])

    def action_start(self) -> None:
        """Start the progress tracking."""

        self.query_one(ProgressBar).update(total=100)
        self.query_one(ProgressBar).percentage = 0
        self.query_one(ProgressBar).advance(1)

    def action_show_info(self) -> None:
        self.query_one(TabbedContent).active = "info"

    def action_show_params(self) -> None:
        self.query_one(TabbedContent).active = "params"


def main():
    args = MonitorApp.parse_args()
    app = MonitorApp(args)
    app.run()
