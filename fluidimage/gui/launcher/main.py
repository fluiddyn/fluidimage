"""Topology launcher Qt GUI (:mod:`fluidimage.gui.launcher.main`)

.. autoclass:: Program
   :members:
   :private-members:

"""

import sys

from matplotlib.backends.qt_compat import QtWidgets

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util.paramcontainer_gui import QtParamContainer
from fluidimage.topologies.launcher import (
    TopologyPIVLauncher,
    TopologyPreprocLauncher,
)

from .mainwindow import Ui_MainWindow


class Program(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        topologies = [TopologyPreprocLauncher, TopologyPIVLauncher]
        self.topologies = {cls.__name__: cls for cls in topologies}

        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        topo_names = ["TopologyPreprocLauncher", "TopologyPIVLauncher"]
        self.actions = {}

        for topo_name in topo_names:
            action = self.actions[topo_name] = QtWidgets.QAction(self)
            self.menuTopologies.addAction(action)
            action.setText(topo_name)

            def func(_, topo_name=topo_name):
                self.init_topo(topo_name)

            action.triggered.connect(func)

        self.actionOpen.triggered.connect(self.open_file)

    def closeEvent(self, QCloseEvent):
        pass

    def open_file(self):
        path_file = QtWidgets.QFileDialog.getOpenFileName(
            self.centralwidget, "OpenFile"
        )

        params = ParamContainer(path_file=path_file)
        print(params)
        raise NotImplementedError

    def init_topo(self, topo_name):
        Topology = self.topologies[topo_name]
        params = Topology.create_default_params()

        self.qt_params = QtParamContainer(
            params, top=True, module_run_from_xml="fluidimage.run_from_xml"
        )

        # first remove all widgets from verticalLayout_2
        layout = self.verticalLayout_2
        for i in reversed(range(layout.count())):
            widgetToRemove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

        self.verticalLayout_2.addWidget(self.qt_params.page_main)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Program()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
