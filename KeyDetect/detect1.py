from PyQt5.QtWidgets import QWidget, QApplication


class Detect(QWidget):
    
    def __init__(self):
        super().__init__()
        self._widget = QWidget(self)
        self.resize(500, 500)
        self._widget.resize(500, 500)
        
    def keyPressEvent(self, a0):
        print("pressed", a0.key())
        return super().keyPressEvent(a0)
    
    def keyReleaseEvent(self, a0):
        print("released", a0.key())
        return super().keyReleaseEvent(a0)
    

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = Detect()
    win.show()
    app.exec()
