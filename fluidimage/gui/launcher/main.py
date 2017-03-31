#!/usr/bin/env python

from __future__ import division

import sys
from copy import deepcopy, copy
import subprocess

from PyQt4 import QtGui

from .mainwindow import Ui_MainWindow, _translate, _fromUtf8

from fluiddyn.util import time_as_str

from fluidimage.topologies.pre_proc import TopologyPreproc
from fluidimage.topologies.piv import TopologyPIV

topologies = [TopologyPreproc, TopologyPIV]
topologies = {cls.__name__: cls for cls in topologies}


class QtParamContainer(object):

    def __init__(self, params, top=False,
                 module_run_from_xml='fluidimage.run_from_xml'):

        self.params = deepcopy(params)
        self.module_run_from_xml = module_run_from_xml
        full_tag_dot = params._make_full_tag()
        self.full_tag = full_tag_dot.replace('.', '_')

        self.labels = {}
        self.lines_edit = {}
        self.buttons = {}

        self.qt_params_children = {}

        key_attribs = params._get_key_attribs()
        tag_children = params._tag_children

        self.page_main = QtGui.QWidget()
        self.page_main.setObjectName(_fromUtf8(
            'page_main_' + self.full_tag))

        self.verticalLayout = QtGui.QVBoxLayout(self.page_main)
        self.verticalLayout.setMargin(0)
        self.verticalLayout.setObjectName(_fromUtf8(
            'verticalLayout_' + self.full_tag))

        if self.params._contains_doc():
            self.pushButton_doc = QtGui.QPushButton(self.page_main)
            self.pushButton_doc.setText('Display doc ' + full_tag_dot)
            self.verticalLayout.addWidget(self.pushButton_doc)
            self.pushButton_doc.released.connect(self.params._print_docs)

        if len(key_attribs) > 0:
            self.page_attribs = QtGui.QWidget()
            self.page_attribs.setObjectName(_fromUtf8(
                'page_attribs_' + self.full_tag))

            self.formLayout_attribs = QtGui.QFormLayout(self.page_attribs)
            self.formLayout_attribs.setMargin(0)
            self.formLayout_attribs.setObjectName(_fromUtf8(
                'formLayout_' + self.full_tag))

            i = -1
            for key in key_attribs:
                i += 1
                label = self.labels[key] = QtGui.QLabel(self.page_attribs)
                label.setObjectName(_fromUtf8(
                    'label_' + self.full_tag + '_' + key))
                self.formLayout_attribs.setWidget(
                    i, QtGui.QFormLayout.LabelRole, label)

                line = self.lines_edit[key] = QtGui.QLineEdit(
                    self.page_attribs)
                line.setObjectName(_fromUtf8(
                    'line_' + self.full_tag + '_' + key))
                self.formLayout_attribs.setWidget(
                    i, QtGui.QFormLayout.FieldRole, line)

                label.setText(_translate('MainWindow', key, None))

                line.setText(_translate(
                    'MainWindow', repr(self.params[key]), None))

                if key == 'path':
                    i += 1

                    def choose_name():
                        fileName = QtGui.QFileDialog.getOpenFileName(
                            self.page_attribs, 'OpenFile')
                        self.lines_edit['path'].setText("'" + fileName + "'")

                    self.buttons[key] = QtGui.QPushButton(self.page_attribs)
                    self.formLayout_attribs.setWidget(
                        i, QtGui.QFormLayout.FieldRole, self.buttons[key])
                    self.buttons[key].setText('Navigate to choose the path')
                    self.buttons[key].released.connect(choose_name)

            self.verticalLayout.addWidget(self.page_attribs)

        if len(tag_children) > 0:
            self.toolBox = QtGui.QToolBox(self.page_main)

            for tag in tag_children:
                qtparam = self.qt_params_children[tag] = self.__class__(
                    self.params[tag])

                self.toolBox.addItem(
                    qtparam.page_main, _fromUtf8(full_tag_dot + '.' + tag))

            self.verticalLayout.addWidget(self.toolBox)

        if top:
            self.pushButton_xml = QtGui.QPushButton(self.page_main)
            self.pushButton_xml.setText('Display as xml')
            self.verticalLayout.addWidget(self.pushButton_xml)
            self.pushButton_xml.released.connect(self.print_as_xml)

            self.pushButton_default = QtGui.QPushButton(self.page_main)
            self.pushButton_default.setText('Reset to default parameters')
            self.verticalLayout.addWidget(self.pushButton_default)
            self.pushButton_default.released.connect(self.reset_default_values)

            self.pushButton_launch = QtGui.QPushButton(self.page_main)
            self.pushButton_launch.setText('Launch computation')
            self.verticalLayout.addWidget(self.pushButton_launch)
            self.pushButton_launch.released.connect(self.launch)

    def reset_default_values(self):
        for key in self.params._get_key_attribs():
            self.lines_edit[key].setText(_translate(
                'MainWindow', repr(self.params[key]), None))

        for tag in self.params._tag_children:
            self.qt_params_children[tag].reset_default_values()

    def produce_params(self):
        params = deepcopy(self.params)
        self.modif_params(params)
        return params

    def print_as_xml(self):
        params = self.produce_params()
        params._print_as_xml()

    def modif_params(self, params):
        for key in self.params._get_key_attribs():
            params[key] = eval(str(self.lines_edit[key].displayText()))

        for tag in self.params._tag_children:
            self.qt_params_children[tag].modif_params(params[tag])

    def launch(self):
        params = self.produce_params()
        d = eval(params._value_text)
        print(d)
        path = params._save_as_xml(
            path_file=params._tag + time_as_str() + '.xml',
            find_new_name=True)
        print(path)
        retcode = subprocess.call(
            ['python', '-m', self.module_run_from_xml, path])
        return retcode


class Program(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

        topo_names = ['TopologyPreproc', 'TopologyPIV']
        self.actions = {}

        for topo_name in topo_names:
            action = self.actions[topo_name] = QtGui.QAction(self)
            self.menuTopologies.addAction(action)
            action.setText(topo_name)

            def func(_, topo_name=topo_name):
                self.init_topo(topo_name)

            action.triggered.connect(func)

    def closeEvent(self, QCloseEvent):
        pass

    def init_topo(self, topo_name):
        Topology = topologies[topo_name]
        params = Topology.create_default_params()

        self.qt_params = QtParamContainer(
            params, top=True, module_run_from_xml='fluidimage.run_from_xml')

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
    app = QtGui.QApplication(sys.argv)
    w = Program()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
