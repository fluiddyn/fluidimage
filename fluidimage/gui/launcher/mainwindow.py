# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(775, 901)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 775, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTopologies = QtWidgets.QMenu(self.menubar)
        self.menuTopologies.setObjectName("menuTopologies")
        self.menuComputations = QtWidgets.QMenu(self.menubar)
        self.menuComputations.setObjectName("menuComputations")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionTopologyPIV = QtWidgets.QAction(MainWindow)
        self.actionTopologyPIV.setObjectName("actionTopologyPIV")
        self.actionTopologyPreproc = QtWidgets.QAction(MainWindow)
        self.actionTopologyPreproc.setObjectName("actionTopologyPreproc")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTopologies.menuAction())
        self.menubar.addAction(self.menuComputations.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTopologies.setTitle(_translate("MainWindow", "Topologies"))
        self.menuComputations.setTitle(_translate("MainWindow", "Computations"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionTopologyPIV.setText(_translate("MainWindow", "TopologyPIV"))
        self.actionTopologyPreproc.setText(
            _translate("MainWindow", "TopologyPreproc")
        )
