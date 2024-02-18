# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.resize(775, 901)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionTopologyPIV = QAction(MainWindow)
        self.actionTopologyPIV.setObjectName("actionTopologyPIV")
        self.actionTopologyPreproc = QAction(MainWindow)
        self.actionTopologyPreproc.setObjectName("actionTopologyPreproc")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 775, 24))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTopologies = QMenu(self.menubar)
        self.menuTopologies.setObjectName("menuTopologies")
        self.menuComputations = QMenu(self.menubar)
        self.menuComputations.setObjectName("menuComputations")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTopologies.menuAction())
        self.menubar.addAction(self.menuComputations.menuAction())
        self.menuFile.addAction(self.actionOpen)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "MainWindow", None)
        )
        self.actionOpen.setText(
            QCoreApplication.translate("MainWindow", "Open", None)
        )
        self.actionTopologyPIV.setText(
            QCoreApplication.translate("MainWindow", "TopologyPIV", None)
        )
        self.actionTopologyPreproc.setText(
            QCoreApplication.translate("MainWindow", "TopologyPreproc", None)
        )
        self.menuFile.setTitle(
            QCoreApplication.translate("MainWindow", "File", None)
        )
        self.menuTopologies.setTitle(
            QCoreApplication.translate("MainWindow", "Topologies", None)
        )
        self.menuComputations.setTitle(
            QCoreApplication.translate("MainWindow", "Computations", None)
        )

    # retranslateUi
