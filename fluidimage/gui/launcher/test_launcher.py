import os

from warnings import warn

GITLAB_CI = os.getenv("GITLAB_CI")

if not GITLAB_CI:

    from .main import Program

    def test_launcher(qtbot):
        widget = Program()
        qtbot.addWidget(widget)

else:
    warn(
        "Skip test in fluidimage.gui.launcher.test_launcher "
        f"because GITLAB_CI={GITLAB_CI}"
    )