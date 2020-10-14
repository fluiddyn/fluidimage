import pytest

# just to be sure that matplotlib is importable
import matplotlib

try:
    import matplotlib.backends.qt_compat
except ImportError:
    import_error_qt_compat = True
else:
    import_error_qt_compat = False

# For Travis where PyQt5 can't be installed (why?)
@pytest.mark.skipif(
    import_error_qt_compat, reason="ImportError matplotlib.backends.qt_compat"
)
def test_launcher(qtbot):
    from .main import Program

    widget = Program()
    qtbot.addWidget(widget)
