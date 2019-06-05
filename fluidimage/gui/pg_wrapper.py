import sys
from pathlib import Path

import numpy as np

from fluidimage.util import imread, logger

try:
    import pyqtgraph as pg
    from pyqtgraph import QtCore, QtGui, dockarea
except ImportError:
    from warnings import warn

    warn(
        "PyQtgraph must be installed to use this module! "
        + "Also one of these must be installed: PyQt>=4.7 / PySide / PyQt5"
    )


class PGWrapper:
    def __init__(self, win_type=None, title="FluidImage"):
        self._create_app()
        self._create_win(win_type, title)
        self._type = win_type
        self.area = None

    def _create_app(self):
        self.app = QtGui.QApplication([])

    def _create_win(self, win_type, title):
        if win_type is None:
            self.win = QtGui.QMainWindow()
            self.win.resize(800, 800)
        elif win_type == "glt_widget":
            self.win = pg.GraphicsLayoutWidget()
        elif win_type == "gfx_window":
            self.win = pg.GraphicsWindow()

        self.win.setWindowTitle(title)

    def _add_dock(self, title="Dock", size=(500, 200), position="bottom"):
        if self.area is None:
            self.area = dockarea.DockArea()
            self.win.setCentralWidget(self.area)

        d = dockarea.Dock(title, size=size)
        self.area.addDock(d, position)
        logger.debug("Function _add_dock: " + repr(d))

    def _add_gfx_item(self, obj, item):
        # GWidget = pg.graphicsItems.GraphicsWidget.GraphicsWidget
        GPlotItem = pg.graphicsItems.GraphicsLayout.PlotItem
        GItem = pg.graphicsItems.GraphicsItem.GraphicsItem

        attr = " "
        if hasattr(obj, "addPlot") and isinstance(item, GPlotItem):
            obj.addPlot(item)
            attr = "addPlot"
        elif hasattr(obj, "addItem") and isinstance(item, GItem):
            obj.addItem(item)
            attr = "addItem"
        elif hasattr(obj, "addWidget"):
            obj.addWidget(item)
            attr = "addWidget"
        elif hasattr(obj, "setCentralWidget"):
            obj.setCentralWidget(item)
            attr = "setCentralWidget"

        logger.debug("Function _add_gfx_item: " + attr + repr(item))

    def _win(self, key=None, attr="__init__"):
        if self.area is None:
            win = self.win
        elif key is None:
            win = self.area.docks.items()[0][1]
        else:
            win = self.area.docks[key]

        logger.debug("Function _win: " + repr(win))
        return win

    def view(self, path, title=None, hide_crosshair=True):
        """
        ImageView, a high-level widget for displaying and analyzing 2D and 3D
        data. ImageView provides:

          1. A zoomable region (ViewBox) for displaying the image
          2. A combination histogram and gradient editor (HistogramLUTItem)
             for controlling the visual appearance of the image
          3. A timeline for selecting the currently displayed frame (for 3D
             data only).
          4. Tools for very basic analysis of image data (see ROI and Norm
             buttons)

        """
        imv = pg.ImageView()
        win = self._win(title)
        self._add_gfx_item(win, imv)

        if not isinstance(path, str):
            data = []
            for p in path:
                logger.info(f"Viewing {p}")
                data.append(imread(p).transpose())

            data = np.array(data)
            imv.setImage(data, xvals=np.linspace(0, len(path), data.shape[0]))
        elif Path(path).is_dir():
            raise ValueError("Expected files not directory.")
        else:
            logger.info(f"Viewing {path}")
            try:
                data = imread(path).transpose()
            except AttributeError:
                raise ValueError(f"Is {path} an image?")
            imv.setImage(data)

        vb = imv.imageItem.getViewBox()
        self._add_crosshair(win, imv, vb, hide_lines=hide_crosshair)

    def _add_crosshair(self, win, p1, vb, hide_lines=True):
        if not hide_lines:
            vLine = pg.InfiniteLine(angle=90, movable=False)
            hLine = pg.InfiniteLine(angle=0, movable=False)
            p1.addItem(vLine, ignoreBounds=True)
            p1.addItem(hLine, ignoreBounds=True)

        label = pg.TextItem("(x,y)=intensity", anchor=(0.0, 0.0))
        p1.addItem(label)
        label.hide()

        def mouseMoved(evt):
            data1 = (
                p1.getImageItem().image
            )  # should be updated using sigTimeChanged
            pos = evt[
                0
            ]  # using signal proxy turns original arguments into a tuple
            mousePoint = vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            index = np.array([x, y], dtype=int)
            index_max = data1.shape
            if np.all(np.greater_equal(index, [0, 0])) and np.all(
                np.less_equal(index, index_max)
            ):
                text = (
                    "<span style='font-size: 10pt; color: cyan'>(%0.1f, %0.1f)=</span>"
                    "<span style='color: red'>%0.1f</span>"
                ) % (mousePoint.x(), mousePoint.y(), data1[index[0], index[1]])
                label.setHtml(text)
                label.setPos(x, y)
                label.show()
                if not hide_lines:
                    vLine.setPos(x)
                    hLine.setPos(y)
            else:
                label.hide()

        try:
            signal = p1.scene.sigMouseMoved
        except AttributeError:
            signal = p1.scene().sigMouseMoved

        self.proxy = pg.SignalProxy(signal, rateLimit=48, slot=mouseMoved)

    def show(self):
        self.win.show()
        if (sys.flags.interactive != 1) or not hasattr(QtCore, "PYQT_VERSION"):
            QtGui.QApplication.instance().exec_()
        else:
            raise ValueError("Cannot start Qt event while in interactive mode")
