# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt4 UI code generator 4.12
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(775, 901)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 24))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuTopologies = QtGui.QMenu(self.menubar)
        self.menuTopologies.setObjectName(_fromUtf8("menuTopologies"))
        self.menuComputations = QtGui.QMenu(self.menubar)
        self.menuComputations.setObjectName(_fromUtf8("menuComputations"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtGui.QAction(MainWindow)
        self.actionOpen.setObjectName(_fromUtf8("actionOpen"))
        self.actionTopologyPIV = QtGui.QAction(MainWindow)
        self.actionTopologyPIV.setObjectName(_fromUtf8("actionTopologyPIV"))
        self.actionTopologyPreproc = QtGui.QAction(MainWindow)
        self.actionTopologyPreproc.setObjectName(_fromUtf8("actionTopologyPreproc"))
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTopologies.menuAction())
        self.menubar.addAction(self.menuComputations.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuTopologies.setTitle(_translate("MainWindow", "Topologies", None))
        self.menuComputations.setTitle(_translate("MainWindow", "Computations", None))
        self.actionOpen.setText(_translate("MainWindow", "Open", None))
        self.actionTopologyPIV.setText(_translate("MainWindow", "TopologyPIV", None))
        self.actionTopologyPreproc.setText(_translate("MainWindow", "TopologyPreproc", None))

