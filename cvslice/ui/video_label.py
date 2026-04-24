"""Custom video label with mouse event forwarding, zoom, and pan."""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF
from PyQt5.QtGui import QWheelEvent, QPainter, QPixmap


class VideoLabel(QLabel):
    """QLabel subclass that emits mouse signals with frame-space coordinates.

    Supports zoom (Ctrl+scroll) and pan (middle-click drag or Ctrl+drag).
    Double middle-click or Ctrl+double-click resets view.
    """

    # Signals: (frame_x, frame_y)
    mouse_pressed = pyqtSignal(int, int)
    mouse_moved = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame_w = 0   # original frame width
        self._frame_h = 0   # original frame height
        # Zoom / pan state
        self._zoom = 1.0
        self._pan = QPointF(0, 0)  # pan offset in widget pixels
        self._panning = False
        self._pan_start = QPoint()
        self._src_pix: QPixmap | None = None  # unscaled source pixmap

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # --- Pixmap handling ---
    def setPixmap(self, pix: QPixmap):
        """Store source pixmap and trigger repaint (no Qt scaling)."""
        self._src_pix = pix
        # Don't call super().setPixmap — we paint manually
        self.update()

    def _base_rect(self) -> QRectF:
        """Compute the base (zoom=1) pixmap rect centered in widget, aspect-fit."""
        if self._src_pix is None or self._src_pix.isNull():
            return QRectF()
        pw, ph = self._src_pix.width(), self._src_pix.height()
        lw, lh = self.width(), self.height()
        scale = min(lw / pw, lh / ph)
        sw, sh = pw * scale, ph * scale
        ox = (lw - sw) / 2.0
        oy = (lh - sh) / 2.0
        return QRectF(ox, oy, sw, sh)

    def paintEvent(self, event):
        if self._src_pix is None or self._src_pix.isNull():
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, not self._zoom > 3.0)
        base = self._base_rect()
        # Apply zoom around center of base rect, then pan
        cx = base.center().x() + self._pan.x()
        cy = base.center().y() + self._pan.y()
        w = base.width() * self._zoom
        h = base.height() * self._zoom
        dst = QRectF(cx - w / 2, cy - h / 2, w, h)
        src = QRectF(0, 0, self._src_pix.width(), self._src_pix.height())
        p.drawPixmap(dst, self._src_pix, src)
        p.end()

    def _to_frame_coords(self, pos: QPoint) -> tuple[int, int] | None:
        """Map widget pixel position to original frame coordinates (zoom+pan aware)."""
        if self._src_pix is None or self._src_pix.isNull():
            return None
        base = self._base_rect()
        if base.width() <= 0 or base.height() <= 0:
            return None
        # Compute zoomed+panned rect (same as paintEvent)
        cx = base.center().x() + self._pan.x()
        cy = base.center().y() + self._pan.y()
        w = base.width() * self._zoom
        h = base.height() * self._zoom
        dst = QRectF(cx - w / 2, cy - h / 2, w, h)
        # Position relative to dst rect → normalized [0,1]
        rx = (pos.x() - dst.x()) / dst.width()
        ry = (pos.y() - dst.y()) / dst.height()
        if rx < 0 or ry < 0 or rx > 1 or ry > 1:
            return None
        if self._frame_w <= 0 or self._frame_h <= 0:
            return None
        fx = int(rx * self._frame_w)
        fy = int(ry * self._frame_h)
        return max(0, min(self._frame_w - 1, fx)), max(0, min(self._frame_h - 1, fy))

    # --- Mouse events ---
    def mousePressEvent(self, event):
        if (event.button() == Qt.MiddleButton or
                (event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier)):
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            coords = self._to_frame_coords(event.pos())
            if coords:
                self.mouse_pressed.emit(*coords)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan += QPointF(delta.x(), delta.y())
            self._pan_start = event.pos()
            self.update()
            return
        coords = self._to_frame_coords(event.pos())
        if coords:
            self.mouse_moved.emit(*coords)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning and (event.button() in (Qt.MiddleButton, Qt.LeftButton)):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if event.button() == Qt.LeftButton:
            coords = self._to_frame_coords(event.pos())
            if coords:
                self.mouse_released.emit(*coords)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            factor = 1.15 if delta > 0 else 1 / 1.15
            new_zoom = max(0.5, min(10.0, self._zoom * factor))
            # Zoom toward cursor: adjust pan so point under cursor stays fixed
            pos = event.position() if hasattr(event, 'position') else QPointF(event.pos())
            base = self._base_rect()
            cx = base.center().x() + self._pan.x()
            cy = base.center().y() + self._pan.y()
            # Point under cursor in "content" space relative to center
            ratio = new_zoom / self._zoom
            self._pan = QPointF(
                self._pan.x() + (pos.x() - cx) * (1 - ratio),
                self._pan.y() + (pos.y() - cy) * (1 - ratio),
            )
            self._zoom = new_zoom
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.modifiers() & Qt.ControlModifier):
            self.reset_view()
            return
        super().mouseDoubleClickEvent(event)
