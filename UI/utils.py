from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

def scale_pixmap(pixmap: QPixmap, width: int, height: int) -> QPixmap:
    """等比缩放到指定矩形内"""
    return pixmap.scaled(width - 10, height - 10,
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
