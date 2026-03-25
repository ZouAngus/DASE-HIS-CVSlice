"""CVSlice entry point."""
import sys
from PyQt5.QtWidgets import QApplication
from cvslice.ui import ClipAnnotator


def main():
    app = QApplication(sys.argv)
    win = ClipAnnotator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
