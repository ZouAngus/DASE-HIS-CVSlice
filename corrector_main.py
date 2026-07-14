"""Standalone entry point for the Skeleton Corrector (PyInstaller target).

Build a single-file Windows exe (from the repo root):

    python -m PyInstaller --noconfirm --clean --onefile --windowed \
        --name CVSlice-SkeletonCorrector corrector_main.py

The exe lands in dist/CVSlice-SkeletonCorrector.exe. It bundles PyQt5,
OpenCV, numpy/pandas/scipy and (if installed at build time) onnxruntime +
rtmlib, so the optional 2D-pose features (camera-guided fill, consistency
check) work too — their models still download once at first use.

Run:  CVSlice-SkeletonCorrector.exe [optional_export_folder]
"""
import os
import sys

# Pre-load native ML runtimes BEFORE PyQt5 (same reason as main.py): on
# Windows, importing onnxruntime after Qt fails with "DLL initialization
# routine failed". Best-effort — without it the 2D-pose features are simply
# unavailable.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtCore import Qt


def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Force a Latin-primary font to avoid full-width digit rendering.
    preferred = ["Segoe UI", "Microsoft YaHei UI", "Arial",
                 "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC"]
    available = set(QFontDatabase().families())
    chosen = "Arial"
    for name in preferred:
        if name in available:
            chosen = name
            break
    app.setFont(QFont(chosen, 9))

    from cvslice.ui.skeleton_corrector import SkeletonCorrector
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    win = SkeletonCorrector(folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
