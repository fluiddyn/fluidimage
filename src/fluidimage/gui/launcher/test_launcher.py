# just to be sure that matplotlib is importable
import matplotlib
import pytest

try:
    import matplotlib.backends.qt_compat
except ImportError:
    import_error_qt_compat = True
else:
    import_error_qt_compat = False


try:
    import pytestqt
except ImportError:
    pytestqt = False


@pytest.mark.skipif(not pytestqt, reason="ImportError pytest-qt")
@pytest.mark.skipif(
    import_error_qt_compat, reason="ImportError matplotlib.backends.qt_compat"
)
def test_launcher(qtbot):
    from .main import Program

    widget = Program()
    qtbot.addWidget(widget)
