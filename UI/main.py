import sys
from PyQt5.QtCore import QPropertyAnimation
from PyQt5.QtWidgets import QApplication
from .widgets import AnimalClassifierApp   # 相对导入

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = AnimalClassifierApp()

    # 启动淡入
    anim = QPropertyAnimation(win, b'windowOpacity')
    anim.setDuration(800)
    anim.setStartValue(0)
    anim.setEndValue(1)
    anim.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()