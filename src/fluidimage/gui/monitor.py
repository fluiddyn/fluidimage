"""Monitor computations done with topologies

.. autoclass:: MonitorApp
   :members:
   :private-members:


"""

import argparse
import os

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Label


class MonitorApp(App):
    """Fluidimage monitor Textual app"""

    TITLE = "Fluidimage monitor app"
    SUB_TITLE = "Monitoring parallel Fluidimage computations"

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
        self.args = args
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Output directory: {self.args.path}")
        yield Footer()


def main():
    args = MonitorApp.parse_args()
    app = MonitorApp(args)
    app.run()
