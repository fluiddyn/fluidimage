"""Monitor computations done with topologies

.. autoclass:: MonitorApp
   :members:
   :private-members:

"""

import argparse
import os
import subprocess
from importlib import import_module
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Middle
from textual.timer import Timer
from textual.widgets import (
    DataTable,
    Digits,
    Footer,
    Header,
    Label,
    Markdown,
    ProgressBar,
    Rule,
    TabbedContent,
    TabPane,
    Tree,
)

from fluidimage import ParamContainer


def build_branch(branch, params_node):
    """Build the branches of the tree"""
    for key in params_node._get_key_attribs():
        branch.add_leaf(f"{key} = {repr(params_node[key])}")

    for tag in params_node._tag_children:
        new_branch = branch.add(tag, expand=False, data=params_node[tag])
        build_branch(new_branch, params_node[tag])


def copy_doc(params_node, params_node_with_doc):
    """Copy the documentation between 2 ParamContainer objects"""
    params_node._set_doc(params_node_with_doc._doc)

    for tag in params_node._tag_children:
        copy_doc(params_node[tag], params_node_with_doc[tag])


class MonitorApp(App):
    """Fluidimage monitor Textual app"""

    progress_bar: ProgressBar
    tree_params: Tree
    timer_update_info: Timer
    digit_num_results: Digits
    widget_doc: Markdown
    job_is_running: bool

    CSS_PATH = "monitor.tcss"

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
        """Parse the arguments of the command line"""

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
        self.check_is_running()

        self.path_info = self.path_job_info / "info.xml"
        info = ParamContainer(path_file=self.path_info)
        self.info_job = {
            key: info[key] for key in sorted(info._get_key_attribs())
        }

        self.params = ParamContainer(path_file=self.path_job_info / "params.xml")

        module_name, class_name = self.info_job["topology"].rsplit(".", 1)
        mod = import_module(module_name)
        class_topology = getattr(mod, class_name)
        params_default = class_topology.create_default_params()
        copy_doc(self.params, params_default)

        self.paths_len_results = sorted(
            self.path_job_info.glob("len_results_*.txt")
        )
        assert self.paths_len_results

        self.num_results = 0
        self.detect_results()

    def detect_results(self):
        """Detect how many results have been computed"""
        num_results_vs_idx_process = []
        for path_len_results in self.paths_len_results:
            with open(path_len_results, encoding="utf-8") as file:
                content = file.read()
                len_results = int(content) if content else 0
                num_results_vs_idx_process.append(len_results)
        self.num_results = sum(num_results_vs_idx_process)

    def check_is_running(self):
        """Check if the executor is running"""
        self.job_is_running = self.path_lockfile.exists()
        return self.job_is_running

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Info", id="info"):
                with Middle():
                    with Center():
                        yield Label(f"Output directory: {str(self.path_results)}")
                        self.label_is_running = Label(
                            f"Running: {self.job_is_running}"
                        )
                        yield self.label_is_running
                    yield Rule()
                    yield DataTable()
                    yield Rule()
                    with Center():
                        yield ProgressBar()
                    with Center():
                        self.digit_num_results = Digits(f"{self.num_results}")
                        yield Horizontal(
                            self.digit_num_results,
                            Label("results", id="label_result"),
                            id="num_results",
                        )

            with TabPane("Parameters", id="params"):
                with Middle():
                    with Horizontal():
                        self.tree_params = Tree("params", data=self.params)
                        self.tree_params.root.expand()
                        build_branch(self.tree_params.root, self.params)
                        yield self.tree_params
                        self.widget_doc = Markdown(self.params._doc)
                        yield self.widget_doc

        yield Footer()

    def on_mount(self) -> None:
        """on_mount Textual method"""
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

        self.progress_bar = self.query_one(ProgressBar)
        self.progress_bar.update(total=self.info_job["num_expected_results"])
        self.progress_bar.progress = self.num_results

        topology_name = self.info_job["topology"].rsplit(".")[-1]
        if topology_name in ("TopologyPIV",):
            self.bind(
                "f", "launch_fluidpivviewer", description="Launch fluidpivviewer"
            )

        self.timer_update_info = self.set_interval(
            2.0, callback=self.update_info, name="update_info"
        )
        if not self.job_is_running:
            self.timer_update_info.pause()

        self.tree_params.styles.border = ("round", "yellow")
        self.tree_params.styles.width = "1fr"
        self.tree_params.border_title = "Parameters"
        self.tree_params.styles.border_title_align = "center"

        self.widget_doc.styles.border = ("round", "yellow")
        self.widget_doc.styles.width = "1fr"
        self.widget_doc.styles.height = "100%"
        self.widget_doc.border_title = "Documentation"
        self.widget_doc.styles.border_title_align = "center"

    def update_info(self):
        """Update the 'info' panel"""
        self.detect_results()
        self.progress_bar.progress = self.num_results
        self.digit_num_results.update(f"{self.num_results}")
        if not self.check_is_running():
            self.timer_update_info.pause()
            self.label_is_running.update(f"Running: {self.job_is_running}")

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

    def on_tree_node_selected(self, event):
        """Triggered when the tree_params is selected"""
        params_node = event.node.data
        if params_node is None:
            return
        self.widget_doc.update(params_node._doc)


def main():
    """Main function for fluidimage-monitor"""
    args = MonitorApp.parse_args()
    app = MonitorApp(args)
    app.run()
