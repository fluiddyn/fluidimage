from .main import Program


def test_launcher(qtbot):
    widget = Program()
    qtbot.addWidget(widget)
