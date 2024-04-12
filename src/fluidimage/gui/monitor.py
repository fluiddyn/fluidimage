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
    for key in params_node._get_key_attribs():
        branch.add_leaf(f"{key} = {repr(params_node[key])}")

    for tag in params_node._tag_children:
        new_branch = branch.add(tag, expand=False)
        build_branch(new_branch, params_node[tag])


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
        self.path_results = Path(self.args.path)

        if not self.path_results.exists():
            print(f"{self.args.path} does not exist.")
            self.exit(0)
            return

        try:
            self.path_job_info = sorted(self.path_results.glob("job_*"))[-1]
        except IndexError:
            print("No job info folder found.")
            self.exit(0)
            return

        self.path_lockfile = self.path_job_info / "is_running.lock"
        self.job_is_running = self.path_lockfile.exists()

        self.path_info = self.path_job_info / "info.xml"
        info = ParamContainer(path_file=self.path_info)
        self.info_job = {
            key: info[key] for key in sorted(info._get_key_attribs())
        }

        self.params = ParamContainer(path_file=self.path_job_info / "params.xml")

        paths_len_results = sorted(self.path_job_info.glob("len_results_*.txt"))

        num_results_vs_idx_process = []
        for path_len_results in paths_len_results:
            with open(path_len_results, encoding="utf-8") as file:
                content = file.read()
                len_results = int(content) if content else 0
                num_results_vs_idx_process.append(len_results)
        self.num_results = sum(num_results_vs_idx_process)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Info", id="info"):
                with Middle():
                    with Center():
                        yield Label(f"Output directory: {str(self.path_results)}")
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
                        self.tree_params.root.expand()
                        build_branch(self.tree_params.root, self.params)
                        yield self.tree_params

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Parameter", "Value")

        keys = [
            "topology",
            "executor",
            "nb_cpus_allowed",
            "nb_max_workers",
            "num_expected_results",
        ]
        lines = [(key, self.info_job[key]) for key in keys]
        table.add_rows(lines)

        progress_bar = self.query_one(ProgressBar)
        progress_bar.update(total=self.info_job["num_expected_results"])
        progress_bar.progress = self.num_results

        topology_name = self.info_job["topology"].rsplit(".")[-1]
        if topology_name in ("TopologyPIV",):
            self.bind(
                "f",
                "launch_fluidpivviewer",
                description="Launch fluidpivviewer",
            )

    def action_launch_fluidpivviewer(self) -> None:
        """Launch fluidpivviewer from the result directory"""
        print("launching fluidpivviewer")
        subprocess.run(["fluidpivviewer", str(self.path_results)], check=False)

    def action_show_info(self) -> None:
        """Show the 'info' panel"""
        self.query_one(TabbedContent).active = "info"

    def action_show_params(self) -> None:
        """Show the 'params' panel"""
        self.query_one(TabbedContent).active = "params"


def main():
    args = MonitorApp.parse_args()
    app = MonitorApp(args)
    app.run()
