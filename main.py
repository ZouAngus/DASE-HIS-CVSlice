"""CVSlice entry point."""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtCore import Qt
from cvslice.ui import ClipAnnotator


def main():
    # High-DPI support
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Force a Latin-primary font to avoid full-width digit rendering
    preferred = ["Segoe UI", "Microsoft YaHei UI", "Arial",
                 "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC"]
    available = set(QFontDatabase().families())
    chosen = "Arial"  # ultimate fallback
    for name in preferred:
        if name in available:
            chosen = name
            break
    font = QFont(chosen, 9)
    app.setFont(font)

    # Optional: launch the standalone Skeleton Corrector window.
    # Usage:  python main.py --correct [optional_folder_path]
    if "--correct" in sys.argv:
        from cvslice.ui.skeleton_corrector import SkeletonCorrector
        idx = sys.argv.index("--correct")
        folder = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        win = SkeletonCorrector(folder)
    else:
        win = ClipAnnotator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
