import os
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFontDatabase, QSurfaceFormat

from UI.UI import Ui_MainWindow


def qt_surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 6)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSamples(4)  # 抗锯齿
    return fmt


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)


if __name__ == "__main__":
    QSurfaceFormat.setDefaultFormat(qt_surface_format())
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()