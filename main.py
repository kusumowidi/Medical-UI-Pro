"""
main.py — Medical UI Pro: Holographic Edition

Gesture-controlled 3D medical STL viewer with webcam hand tracking,
hardware-accelerated OpenGL rendering, collapsible sidebar, color picker,
mesh stats, anatomical view presets, screenshot export, and theme toggle.
"""

import logging
import math
import os
import queue
import sys
import time
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QGroupBox, QSlider, QMessageBox, QComboBox,
    QProgressBar, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor, QShortcut, QKeySequence

from models import load_stl_model, MAX_FILE_SIZE_MB

# ─── Conditional OpenGL ──────────────────────────────────────────────────────
try:
    from gl_renderer import GLViewport, _GL_OK
    HAS_OPENGL = _GL_OK
except ImportError:
    HAS_OPENGL = False

if not HAS_OPENGL:
    from renderer import SoftwareRenderer

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 1280, 720
TIMER_MS = 30
VIEW_SMOOTH = 0.15
PINCH_THRESH = 0.06

DEFAULT_VIEW: Dict[str, float] = {
    "rot_x": 20.0, "rot_y": -30.0, "zoom": 1.0,
    "pan_x": 0.0, "pan_y": 0.0, "xray_density": 0.3,
}

HUD_COLOR = (0, 240, 255)

# Color presets (#9)
COLOR_PRESETS = {
    "Bone (White)":     (0.92, 0.90, 0.85),
    "Tissue (Pink)":    (0.85, 0.55, 0.50),
    "Organ (Red)":      (0.75, 0.25, 0.20),
    "Vessel (Blue)":    (0.30, 0.45, 0.80),
    "Cartilage (Teal)": (0.40, 0.75, 0.70),
    "Metal Implant":    (0.70, 0.72, 0.75),
    "Nerve (Yellow)":   (0.90, 0.85, 0.30),
}

# Anatomical view presets (#16)
VIEW_PRESETS = {
    "ANT":  {"rot_x": 0,   "rot_y": 0},
    "POST": {"rot_x": 0,   "rot_y": 180},
    "L":    {"rot_x": 0,   "rot_y": -90},
    "R":    {"rot_x": 0,   "rot_y": 90},
    "SUP":  {"rot_x": -90, "rot_y": 0},
    "INF":  {"rot_x": 90,  "rot_y": 0},
}

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ─── Gesture Helpers ──────────────────────────────────────────────────────────
def _dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def _lpx(lm, w, h): return int(lm.x * w), int(lm.y * h)

def count_fingers(hl, label):
    lm = hl.landmark
    f = [lm[4].x < lm[3].x] if label == "Right" else [lm[4].x > lm[3].x]
    for t, j in zip([8,12,16,20], [6,10,14,18]):
        f.append(lm[t].y < lm[j].y)
    return f

def classify_gesture(hl, label, w, h):
    f = count_fingers(hl, label)
    n = sum(f)
    nd = _dist(_lpx(hl.landmark[4], w, h), _lpx(hl.landmark[8], w, h)) / w
    if nd < PINCH_THRESH: return "pinch"
    if n >= 4: return "open"
    if n == 0: return "fist"
    if f[1] and not f[2]: return "point"
    if f[1] and f[2] and not f[3]: return "peace"
    return "other"

def hand_center(hl, w, h):
    xs, ys = zip(*[(l.x*w, l.y*h) for l in hl.landmark])
    return int(np.mean(xs)), int(np.mean(ys))


# ─── Interactive Canvas (Software Renderer Fallback) ─────────────────────────
class InteractiveCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.app_ref = None
        self._m_down = False; self._m_btn = None; self._mx = 0; self._my = 0

    def mousePressEvent(self, e):
        b = e.button()
        if b == Qt.MouseButton.LeftButton: self._m_btn = "L"
        elif b == Qt.MouseButton.RightButton: self._m_btn = "R"
        else: return
        self._m_down = True; self._mx = e.pos().x(); self._my = e.pos().y()

    def mouseReleaseEvent(self, e):
        self._m_down = False
        if self.app_ref: self.app_ref.interaction_points.clear()

    def mouseMoveEvent(self, e):
        if not self._m_down or not self.app_ref: return
        x, y = e.pos().x(), e.pos().y()
        dx, dy = x - self._mx, y - self._my
        if self._m_btn == "L":
            self.app_ref.target_view["rot_y"] += dx * 0.4
            self.app_ref.target_view["rot_x"] += dy * 0.4
        elif self._m_btn == "R":
            self.app_ref.target_view["pan_x"] += dx * 0.6
            self.app_ref.target_view["pan_y"] += dy * 0.6
        self._mx, self._my = x, y

    def wheelEvent(self, e):
        if not self.app_ref: return
        z = self.app_ref.target_view["zoom"]
        z *= 1.1 if e.angleDelta().y() > 0 else 0.9
        self.app_ref.target_view["zoom"] = max(0.1, min(10.0, z))


# ─── Hand Tracking Worker ────────────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        self._cap = None; self._running = False; self._thread = None
        self.result_queue = queue.Queue(maxsize=2)
        self.camera_ok = False

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.camera_ok = False; return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.camera_ok = True; self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start(); return True

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2)
        if self._cap: self._cap.release()

    def _run(self):
        hands = mp_hands.Hands(min_detection_confidence=0.7,
                               min_tracking_confidence=0.6, max_num_hands=2)
        while self._running and self._cap:
            ret, frame = self._cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            hd = []
            if res.multi_hand_landmarks:
                for hl, hn in zip(res.multi_hand_landmarks, res.multi_handedness):
                    lb = hn.classification[0].label
                    g = classify_gesture(hl, lb, w, h)
                    cx, cy = hand_center(hl, w, h)
                    hd.append((g, (cx, cy), lb))
            ann = frame.copy()
            ann = cv2.addWeighted(ann, 0.4, np.zeros_like(ann), 0.6, 0)
            if res.multi_hand_landmarks:
                for hl in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(ann, hl, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=HUD_COLOR, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=1))
                for gs, (cx, cy), _ in hd:
                    cv2.putText(ann, f"[{gs.upper()}]", (cx-20, cy-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, HUD_COLOR, 1)
            _hud_brackets(ann, h, w)
            r = {"hand_data": hd, "annotated": ann}
            try: self.result_queue.put_nowait(r)
            except queue.Full:
                try: self.result_queue.get_nowait()
                except queue.Empty: pass
                self.result_queue.put_nowait(r)
        hands.close()


def _hud_brackets(img, h, w):
    L, c = 20, HUD_COLOR
    for (x0,y0),(dx,dy) in [((10,10),(L,0)),((10,10),(0,L)),((w-10,10),(-L,0)),
        ((w-10,10),(0,L)),((10,h-10),(L,0)),((10,h-10),(0,-L)),
        ((w-10,h-10),(-L,0)),((w-10,h-10),(0,-L))]:
        cv2.line(img, (x0,y0), (x0+dx,y0+dy), c, 2)
    cx, cy = w//2, h//2
    cv2.circle(img, (cx,cy), 100, (0,100,150), 1)
    cv2.circle(img, (cx,cy), 4, c, -1)


# ═════════════════════════════════════════════════════════════════════════════
class AppViewer(QMainWindow):
    """Primary application window for Medical UI Pro — Holographic Edition.

    Manages the full application lifecycle:

    - **Hand tracking**: A background :class:`HandTracker` daemon reads webcam
      frames, runs MediaPipe, classifies gestures, and pushes results into a
      thread-safe queue. The GUI timer drains this queue on each tick.

    - **Rendering**: Supports two interchangeable backends:

        * ``GLViewport`` (gl_renderer.py) — hardware OpenGL 3.3 via PyOpenGL.
        * ``SoftwareRenderer`` (renderer.py) — CPU Phong via NumPy + OpenCV,
          activated automatically when PyOpenGL is unavailable.

    - **View state**: A ``target_view`` dict is mutated by gestures, mouse,
      sliders, and keyboard shortcuts. Each tick applies an exponential lerp
      from ``current_view`` toward ``target_view`` for smooth animation.

    - **Two-way slider binding**: Sliders push to ``target_view``; the timer
      pushes ``current_view`` back to sliders via :meth:`_sync_sliders`.

    Signals:
        mesh_loaded (object): Emitted from the STL worker thread with the
            loaded mesh dict. Connected to :meth:`_on_mesh_loaded`.
        load_error (str): Emitted from the STL worker thread on failure.
            Connected to :meth:`_on_load_error`.

    Attributes:
        target_view (dict): Desired view state (mutated by controls).
        current_view (dict): Smoothed view state (used for rendering).
        meshes (dict): ``{0: mesh_dict}`` — currently only one mesh slot.
        tracker (HandTracker): Background hand tracking thread.
        fps (float): Exponential-moving-average frames per second.
    """

    mesh_loaded = pyqtSignal(object)
    load_error = pyqtSignal(str)

    def __init__(self):
        """Initialize the application window, UI, tracker, and timer.

        Construction order:
            1. Configure window title and size.
            2. Load the initial QSS stylesheet (dark mode).
            3. Initialize view state dicts.
            4. Conditionally construct the software renderer.
            5. Initialize gesture tracking state.
            6. Connect signals to GUI-thread slots (for cross-thread safety).
            7. Construct the full UI via :meth:`_init_ui`.
            8. Register keyboard shortcuts via :meth:`_init_shortcuts`.
            9. Start the :class:`HandTracker` (shows warning if no webcam).
            10. Start the 30 ms render/update timer.
            11. Auto-load the default STL (``heart_NIH3D.stl``) if present.
        """
        super().__init__()
        self.setWindowTitle("Medical UI Pro — Holographic Edition")
        self.resize(1280, 800)

        self._dark_mode = True
        self._use_gl = HAS_OPENGL
        self._load_stylesheet("style.qss")

        # View state
        self.target_view = dict(DEFAULT_VIEW)
        self.current_view = {**DEFAULT_VIEW, "is_loading_stl": True, "xray_mode": False}

        # Rendering
        self.render_w, self.render_h = 960, 800
        self.meshes: Dict[int, dict] = {}
        self._mesh_metadata: Optional[dict] = None
        self._rgb_buf = None; self._pip_buf = None

        if not self._use_gl:
            self.sw_renderer = SoftwareRenderer(self.render_w, self.render_h)

        # Gesture state
        self.prev_center = [None, None]; self.prev_pinch_y = None
        self.last_gesture = [None, None]; self.last_hand_time = time.time()
        self.interaction_points = []
        self.fps = 0.0; self._fps_t = time.time()
        self.current_name = "NO DATABANK"

        # Signals
        self.mesh_loaded.connect(self._on_mesh_loaded)
        self.load_error.connect(self._on_load_error)

        # Tracker
        self.tracker = HandTracker()

        self._init_ui()
        self._init_shortcuts()

        if not self.tracker.start():
            QMessageBox.warning(self, "Camera Unavailable",
                "No webcam detected. Mouse-only mode active.")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_loop)
        self._timer.start(TIMER_MS)

        default = self._find_default_stl()
        if default:
            threading.Thread(target=self._stl_worker, args=(default,), daemon=True).start()
        else:
            self.current_view["is_loading_stl"] = False

    # ── Stylesheet ────────────────────────────────────────────────────
    def _load_stylesheet(self, name):
        p = os.path.join(os.path.dirname(__file__), name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    @staticmethod
    def _find_default_stl():
        d = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(d, "heart_NIH3D.stl")
        return p if os.path.isfile(p) else None

    # ── UI Construction ───────────────────────────────────────────────
    def _init_ui(self):
        main_w = QWidget(); self.setCentralWidget(main_w)
        root = QHBoxLayout(main_w)
        root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── LEFT PANEL (collapsible, scrollable) ─────────────────────
        self._sidebar = QWidget()
        self._sidebar.setFixedWidth(280)
        self._sidebar.setStyleSheet("background-color: #070d14;")
        sidebar_outer = QVBoxLayout(self._sidebar)
        sidebar_outer.setContentsMargins(0,0,0,0); sidebar_outer.setSpacing(0)

        # Collapse button (outside scroll area so always visible)
        self._btn_collapse = QPushButton("◀ COLLAPSE")
        self._btn_collapse.setFixedHeight(28)
        self._btn_collapse.clicked.connect(self._toggle_sidebar)
        sidebar_outer.addWidget(self._btn_collapse)

        # Scroll area wrapping all sidebar content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_inner = QWidget()
        sl = QVBoxLayout(scroll_inner)
        sl.setContentsMargins(8,6,8,6); sl.setSpacing(4)

        # PIP camera (compact)
        self.pip_label = QLabel()
        self.pip_label.setMinimumSize(264, 150)
        self.pip_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.pip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pip_label.setStyleSheet("border: 1px solid #00f3ff;")
        lbl_pip = QLabel("[ OPTICAL SENSOR ]")
        lbl_pip.setStyleSheet("color: #00f3ff; font-weight: bold; font-size: 10px;")
        sl.addWidget(lbl_pip); sl.addWidget(self.pip_label)

        # TELEMETRY
        tg = QGroupBox("TELEMETRY"); tl = QVBoxLayout(tg)
        tl.setContentsMargins(6,14,6,6); tl.setSpacing(2)
        self.lbl_gestures = QLabel("SENSORS: STANDBY")
        self.lbl_gestures.setStyleSheet("font-size: 11px;")
        tl.addWidget(self.lbl_gestures)
        w_rx, self.sl_rx = self._mslider("ROT_X", -180, 180, 20, lambda v: self.target_view.update({"rot_x": float(v)}))
        w_ry, self.sl_ry = self._mslider("ROT_Y", -180, 180, -30, lambda v: self.target_view.update({"rot_y": float(v)}))
        w_z, self.sl_z = self._mslider("ZOOM", 10, 500, 100, lambda v: self.target_view.update({"zoom": v/100.0}))
        w_xd, self.sl_xd = self._mslider("DENSITY", 5, 200, 30, lambda v: self.target_view.update({"xray_density": v/100.0}))
        for w in [w_rx, w_ry, w_z, w_xd]: tl.addWidget(w)
        sl.addWidget(tg)

        # ─── Scan Metadata Panel ─────────────────────────────────────────────
        mg = QGroupBox("SCAN METADATA"); ml = QVBoxLayout(mg)
        ml.setContentsMargins(6,14,6,4); ml.setSpacing(1)
        self.lbl_meta_file = QLabel("File: —")
        self.lbl_meta_verts = QLabel("Verts: —  |  Faces: —")
        self.lbl_meta_bbox = QLabel("BBox: —")
        self.lbl_meta_area = QLabel("Area: —")
        self.lbl_meta_vol = QLabel("Vol: —  |  Watertight: —")
        for w in [self.lbl_meta_file, self.lbl_meta_verts, self.lbl_meta_bbox,
                  self.lbl_meta_area, self.lbl_meta_vol]:
            w.setStyleSheet("font-size: 10px;")
            ml.addWidget(w)
        sl.addWidget(mg)

        # ─── Material Color Picker ────────────────────────────────────────────
        cg = QGroupBox("MATERIAL"); cl = QVBoxLayout(cg)
        cl.setContentsMargins(6,14,6,6)
        self.color_combo = QComboBox()
        self.color_combo.addItems(COLOR_PRESETS.keys())
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        cl.addWidget(self.color_combo)
        sl.addWidget(cg)

        # ─── Anatomical View Presets ──────────────────────────────────────────
        vg = QGroupBox("ANATOMICAL VIEWS"); vgl = QVBoxLayout(vg)
        vgl.setContentsMargins(6,14,6,4); vgl.setSpacing(2)
        row1 = QHBoxLayout(); row2 = QHBoxLayout()
        preset_items = list(VIEW_PRESETS.items())
        for i, (name, vals) in enumerate(preset_items):
            b = QPushButton(name)
            b.setProperty("class", "preset")
            b.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            (row1 if i < 3 else row2).addWidget(b)
        vgl.addLayout(row1); vgl.addLayout(row2)
        sl.addWidget(vg)

        # CONTROLS (compact)
        og = QGroupBox("CONTROLS"); ol = QVBoxLayout(og)
        ol.setContentsMargins(6,14,6,4); ol.setSpacing(1)
        for txt in ["[OPEN] Rot  [POINT] Pan  [PINCH] Zoom",
                    "[PEACE] Rst  [L/R DRAG] Rot/Pan",
                    "Ctrl+O Load  X XRay  W Wire  R Reset",
                    "Ctrl+S Shot  Tab Side  F Full"]:
            lb = QLabel(txt); lb.setStyleSheet("font-size: 10px;"); ol.addWidget(lb)
        sl.addWidget(og)

        # COMMANDS
        kg = QGroupBox("COMMANDS"); kl = QVBoxLayout(kg)
        kl.setContentsMargins(6,14,6,6); kl.setSpacing(3)

        self.btn_load = QPushButton("LOAD DATABANK (STL)")
        self.btn_load.clicked.connect(self._action_load)
        kl.addWidget(self.btn_load)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0); self.progress.setVisible(False)
        kl.addWidget(self.progress)

        self.btn_xray = QPushButton("TOGGLE X-RAY")
        self.btn_xray.setCheckable(True)
        self.btn_xray.clicked.connect(self._action_xray)
        kl.addWidget(self.btn_xray)

        self.btn_wire = QPushButton("TOGGLE WIREFRAME")
        self.btn_wire.setCheckable(True)
        self.btn_wire.clicked.connect(self._action_wireframe)
        kl.addWidget(self.btn_wire)

        btn_reset = QPushButton("RESET VIEW")
        btn_reset.clicked.connect(self._action_reset)
        kl.addWidget(btn_reset)

        btn_shot = QPushButton("CAPTURE VIEWPORT")
        btn_shot.clicked.connect(self._action_screenshot)
        kl.addWidget(btn_shot)

        btn_theme = QPushButton("TOGGLE THEME")
        btn_theme.clicked.connect(self._action_theme)
        kl.addWidget(btn_theme)

        sl.addWidget(kg)

        scroll_inner.setLayout(sl)
        scroll.setWidget(scroll_inner)
        sidebar_outer.addWidget(scroll)

        # ── EXPAND BUTTON (shown when sidebar collapsed) ─────────────
        self._btn_expand = QPushButton("▶")
        self._btn_expand.setFixedWidth(30)
        self._btn_expand.clicked.connect(self._toggle_sidebar)
        self._btn_expand.setVisible(False)

        # ── 3D VIEWPORT ──────────────────────────────────────────────
        if self._use_gl:
            self.viewport = GLViewport(self)
            self.viewport.app_ref = self
            self.viewport.setStyleSheet("border-left: 1px solid #005f73;")
        else:
            self.viewport = InteractiveCanvas(self)
            self.viewport.app_ref = self
            self.viewport.setStyleSheet("background-color: #05080f; border-left: 1px solid #005f73;")
            self.viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)

        root.addWidget(self._btn_expand)
        root.addWidget(self._sidebar)
        root.addWidget(self.viewport, stretch=1)

        self.status = self.statusBar()
        self.status.showMessage(">>> INITIALIZING NEURAL ENGINE...")

        renderer_label = "OpenGL" if self._use_gl else "Software"
        logger.info("Renderer: %s", renderer_label)

    def _mslider(self, label, mn, mx, val, slot):
        row = QWidget(); lay = QHBoxLayout(row)
        lay.setContentsMargins(0,3,0,3)
        lbl = QLabel(label); lbl.setFixedWidth(70)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(mn, mx); s.setValue(val)
        s.valueChanged.connect(slot)
        lay.addWidget(lbl); lay.addWidget(s)
        return row, s

    # ── Keyboard Shortcuts (#13) ──────────────────────────────────────
    def _init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self, self._action_load)
        QShortcut(QKeySequence("Ctrl+S"), self, self._action_screenshot)
        QShortcut(QKeySequence("X"), self, self._action_xray)
        QShortcut(QKeySequence("W"), self, self._action_wireframe)
        QShortcut(QKeySequence("R"), self, self._action_reset)
        QShortcut(QKeySequence("F"), self, self._toggle_fullscreen)
        QShortcut(QKeySequence("Tab"), self, self._toggle_sidebar)

    # ── Actions ──────────────────────────────────────────────────────
    def _toggle_sidebar(self):  # Collapse/expand sidebar (Tab or collapse button)
        vis = not self._sidebar.isVisible()
        self._sidebar.setVisible(vis)
        self._btn_expand.setVisible(not vis)

    def _toggle_fullscreen(self):
        if self.isFullScreen(): self.showNormal()
        else: self.showFullScreen()

    def _action_load(self):
        if self.current_view.get("is_loading_stl"): return
        fp, _ = QFileDialog.getOpenFileName(self, "Select 3D Medical Scan", "", "STL Files (*.stl)")
        if not fp: return
        sz = os.path.getsize(fp) / (1024*1024)
        if sz > MAX_FILE_SIZE_MB:
            r = QMessageBox.question(self, "Large File",
                f"File is {sz:.0f} MB.\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes: return
        self.current_view["is_loading_stl"] = True
        self.btn_load.setEnabled(False)
        self.progress.setVisible(True)
        if self._use_gl: self.viewport.set_loading(True)
        threading.Thread(target=self._stl_worker, args=(fp,), daemon=True).start()

    def _action_xray(self):
        self.current_view["xray_mode"] = not self.current_view.get("xray_mode", False)
        self.btn_xray.setChecked(self.current_view["xray_mode"])
        if self.current_view["xray_mode"] and self.btn_wire.isChecked():
            self.btn_wire.setChecked(False)
            if self._use_gl: self.viewport.set_wireframe(False)

    def _action_wireframe(self):
        on = self.btn_wire.isChecked()
        if self._use_gl: self.viewport.set_wireframe(on)
        if on and self.btn_xray.isChecked():
            self.btn_xray.setChecked(False)
            self.current_view["xray_mode"] = False

    def _action_reset(self):
        self.target_view.update(dict(DEFAULT_VIEW))

    def _action_screenshot(self):  # Screenshot export (Ctrl+S)
        path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "capture.png", "PNG (*.png)")
        if not path: return
        if self._use_gl:
            img = self.viewport.grab_screenshot()
            img.save(path)
        elif self._rgb_buf is not None:
            cv2.imwrite(path, cv2.cvtColor(self._rgb_buf, cv2.COLOR_RGB2BGR))
        logger.info("Screenshot saved: %s", path)

    def _action_theme(self):  # Dark/light theme toggle
        self._dark_mode = not self._dark_mode
        if self._dark_mode:
            self._load_stylesheet("style.qss")
            self._sidebar.setStyleSheet("background-color: #070d14;")
            if self._use_gl: self.viewport.set_bg(0.02, 0.03, 0.06)
            else:
                self.sw_renderer = SoftwareRenderer(self.render_w, self.render_h, dark=True)
        else:
            self._load_stylesheet("style_light.qss")
            self._sidebar.setStyleSheet("background-color: #e8ecf0;")
            if self._use_gl: self.viewport.set_bg(0.92, 0.93, 0.95)
            else:
                self.sw_renderer = SoftwareRenderer(self.render_w, self.render_h, dark=False)

    def _on_color_changed(self, name):  # Material color picker callback
        c = COLOR_PRESETS.get(name)
        if c is None: return
        if 0 in self.meshes:
            self.meshes[0]["color"] = c
        if self._use_gl:
            self.viewport.set_color(c)

    def _apply_preset(self, vals):  # Anatomical view preset callback
        self.target_view.update({"rot_x": float(vals["rot_x"]),
                                 "rot_y": float(vals["rot_y"]),
                                 "zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0})

    # ── STL Loading ──────────────────────────────────────────────────
    def _stl_worker(self, fp: str) -> None:
        """Background thread target: load an STL file and emit the result signal.

        Runs on a daemon thread. Calls load_stl_model() (which is CPU-bound
        and may take several seconds for large files). On success, emits
        ``mesh_loaded`` to be handled on the GUI thread via Qt signal dispatch.

        Thread safety:
            - This method must NOT touch any QWidget or QMainWindow attributes.
            - All UI updates must go through the ``mesh_loaded`` / ``load_error``
              signals which are connected to GUI-thread slots.

        Args:
            fp: Absolute file path to the ``.stl`` file to load.
        """
        mesh = load_stl_model(fp)
        if mesh: self.mesh_loaded.emit(mesh)
        else: self.load_error.emit(f"Failed to load:\n{os.path.basename(fp)}")

    def _on_mesh_loaded(self, mesh):
        self.meshes[0] = mesh
        self.current_view["is_loading_stl"] = False
        self.btn_load.setEnabled(True)
        self.progress.setVisible(False)

        # Reset view to default so new model is properly framed
        self.target_view.update(dict(DEFAULT_VIEW))

        if self._use_gl:
            self.viewport.set_mesh(mesh)
            self.viewport.set_loading(False)
            c = COLOR_PRESETS.get(self.color_combo.currentText(), (0.85,0.85,0.85))
            self.viewport.set_color(c)
            mesh["color"] = c
        self._mesh_metadata = mesh.get("metadata")
        self._update_metadata_ui()
        nv, nf = len(mesh["verts"]), len(mesh["faces"])
        self.current_name = f"{nv} V / {nf} F"
        logger.info("Mesh activated: %s", self.current_name)

    def _on_load_error(self, msg):
        self.current_view["is_loading_stl"] = False
        self.btn_load.setEnabled(True)
        self.progress.setVisible(False)
        if self._use_gl: self.viewport.set_loading(False)
        QMessageBox.critical(self, "STL Error", msg)

    def _update_metadata_ui(self) -> None:  # Scan metadata panel refresh
        m = self._mesh_metadata
        if not m: return
        self.lbl_meta_file.setText(f"File: {m['filename']}")
        self.lbl_meta_verts.setText(f"Verts: {m['original_verts']:,}  |  Faces: {m['original_faces']:,}")
        bb = m.get("bbox_size", [0,0,0])
        self.lbl_meta_bbox.setText(f"BBox: {bb[0]:.1f} × {bb[1]:.1f} × {bb[2]:.1f}")
        self.lbl_meta_area.setText(f"Area: {m.get('surface_area', 0):.2f}")
        vol = m.get("volume")
        wt = "Yes" if m.get("is_watertight") else "No"
        self.lbl_meta_vol.setText(f"Vol: {vol:.2f}" if vol else f"Vol: N/A  |  Watertight: {wt}")

    # ── Slider Sync ──────────────────────────────────────────────────
    def _sync_sliders(self) -> None:
        """Push current_view values back into the sidebar sliders (two-way binding).

        Called every tick from :meth:`_tick` after the exponential lerp so that
        the sliders always reflect the live view state, not just user input.

        ``blockSignals(True)`` prevents the slider's ``valueChanged`` signal from
        firing during the programmatic update, which would create a feedback loop
        causing jitter in gesture-driven motion.
        """
        for s, k, lo, hi, sc in [
            (self.sl_rx, "rot_x", -180, 180, 1), (self.sl_ry, "rot_y", -180, 180, 1),
            (self.sl_z, "zoom", 10, 500, 100), (self.sl_xd, "xray_density", 5, 200, 100)]:
            s.blockSignals(True)
            s.setValue(int(np.clip(self.current_view[k] * sc, lo, hi)))
            s.blockSignals(False)

    # ── Gesture Processing ────────────────────────────────────────────
    def _process_gestures(self, hd: list) -> None:
        """Process hand landmark data and update the target view state.

        Implements a per-frame, stateful gesture state machine. Continuity
        between frames is maintained via ``prev_center`` (for rotate/pan)
        and ``prev_pinch_y`` (for pinch-zoom). Stale state is cleared when
        no hands are detected for > 400 ms.

        Args:
            hd: List of ``(gesture_str, (cx, cy), label)`` tuples from HandTracker.
                Len 0 = no hands; len 1 = single hand; len 2 = dual hand.

        Side effects:
            Mutates ``self.target_view`` keys: rot_x, rot_y, zoom, pan_x, pan_y.
            Mutates ``self.interaction_points`` for HUD dot rendering.
        """
        if not (hasattr(self.viewport, '_m_down') and self.viewport._m_down):
            self.interaction_points.clear()
        n = len(hd)
        if n == 0:
            if time.time() - self.last_hand_time > 0.4:
                self.prev_center = [None, None]; self.prev_pinch_y = None
            self.last_gesture = [None, None]; return
        self.last_hand_time = time.time()
        self.last_gesture = [hd[i][0] if i < n else None for i in range(2)]
        if n == 1:
            g, (cx, cy), _ = hd[0]
            if g in ("open","pinch","point"): self.interaction_points.append((cx,cy))
            if g == "open":
                if self.prev_center[0]:
                    self.target_view["rot_y"] += (cx - self.prev_center[0][0]) * 0.5
                    self.target_view["rot_x"] += (cy - self.prev_center[0][1]) * 0.5
                self.prev_center[0] = (cx, cy); self.prev_pinch_y = None
            elif g == "pinch":
                if self.prev_pinch_y is not None:
                    self.target_view["zoom"] -= (cy - self.prev_pinch_y) * 0.008
                    self.target_view["zoom"] = max(0.1, min(10, self.target_view["zoom"]))
                self.prev_pinch_y = cy; self.prev_center[0] = None
            elif g == "point":
                if self.prev_center[0]:
                    self.target_view["pan_x"] += (cx - self.prev_center[0][0]) * 0.8
                    self.target_view["pan_y"] += (cy - self.prev_center[0][1]) * 0.8
                self.prev_center[0] = (cx, cy); self.prev_pinch_y = None
            elif g == "peace":
                self.target_view.update(dict(DEFAULT_VIEW))
                self.prev_center = [None, None]; self.prev_pinch_y = None
            else:
                self.prev_center[0] = None; self.prev_pinch_y = None
        elif n == 2:
            _, (cx0,cy0), _ = hd[0]; _, (cx1,cy1), _ = hd[1]
            ax, ay = (cx0+cx1)//2, (cy0+cy1)//2
            if hd[0][0] == "open" and hd[1][0] == "open":
                self.interaction_points.extend([(cx0,cy0),(cx1,cy1)])
                if self.prev_center[1]:
                    self.target_view["pan_x"] += (ax - self.prev_center[1][0]) * 0.8
                    self.target_view["pan_y"] += (ay - self.prev_center[1][1]) * 0.8
                self.prev_center[1] = (ax, ay)
            else: self.prev_center[1] = None

    # ── Main Update Loop ─────────────────────────────────────────────
    def _update_loop(self):
        try: self._tick()
        except Exception as e: logger.exception("Update error: %s", e)

    def _tick(self) -> None:
        """Single frame update: consume gesture queue, lerp view, render, update UI.

        Called every 30 ms by the QTimer. Order of operations:
            1. Drain the HandTracker queue (latest frame wins).
            2. Run the gesture state machine on hand data.
            3. Apply exponential lerp: current_view ← current_view + Δ * VIEW_SMOOTH.
            4. Push current_view back to sliders (:meth:`_sync_sliders`).
            5. Dispatch to GL or software renderer.
            6. Update the PIP camera overlay.
            7. Update FPS counter (exponential moving average, α=0.1).
            8. Update status bar with FPS, mesh info, mode, camera, engine.
        """
        # ─── Hand tracking: drain queue (latest frame wins) ───────────────
        hd, ann = [], None
        while not self.tracker.result_queue.empty():
            try:
                r = self.tracker.result_queue.get_nowait()
                hd, ann = r["hand_data"], r["annotated"]
            except queue.Empty: break
        self._process_gestures(hd)

        # Smooth view
        for k in self.target_view:
            cv = self.current_view.get(k, self.target_view[k])
            self.current_view[k] = cv + (self.target_view[k] - cv) * VIEW_SMOOTH
        self._sync_sliders()

        # ── GL Rendering Path ─────────────────────────────────────────
        if self._use_gl:
            self.viewport.set_view(self.current_view)
            self.viewport.set_fps(self.fps)
            self.viewport.update()
        else:
            # ── Software Rendering Path ────────────────────────────────
            cw, ch = self.viewport.width(), self.viewport.height()
            if cw > 0 and ch > 0 and (cw != self.render_w or ch != self.render_h):
                self.render_w, self.render_h = cw, ch
                dark = self._dark_mode
                self.sw_renderer = SoftwareRenderer(self.render_w, self.render_h, dark=dark)
            panel = np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)
            mesh = self.meshes.get(0, {"verts": np.zeros((0,3), np.float32),
                "faces": np.zeros((0,3), np.int32), "color": (0.85,0.85,0.85), "style": "solid"})
            self.sw_renderer.render(panel, mesh, self.current_view)
            if self.current_view.get("is_loading_stl"):
                t = time.time()
                dots = "." * (int(t * 3) % 4)
                # Spinner arc
                cx, cy = self.render_w // 2, self.render_h // 2
                angle = int(t * 360) % 360
                cv2.ellipse(panel, (cx, cy), (35, 35), 0, angle, angle + 270, (0, 243, 255), 3)
                cv2.putText(panel, f"LOADING DATABANK{dots}", (cx - 120, cy + 65),
                            cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 100, 200), 1)
            if panel.shape[0] > 0 and panel.shape[1] > 0:
                h, w, ch = panel.shape
                self._rgb_buf = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
                qi = QImage(self._rgb_buf.data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.viewport.setPixmap(QPixmap.fromImage(qi))

        # Mesh name
        mesh = self.meshes.get(0)
        if mesh and len(mesh["verts"]) > 0:
            self.current_name = f"{len(mesh['verts'])} V / {len(mesh['faces'])} F"
        else:
            self.current_name = "NO DATABANK"

        # PIP camera
        if ann is not None:
            ph, pw = ann.shape[:2]
            self._pip_buf = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            qi = QImage(self._pip_buf.data, pw, ph, pw*3, QImage.Format.Format_RGB888)
            lw, lh = self.pip_label.width(), self.pip_label.height()
            qi = qi.scaled(lw, lh, Qt.AspectRatioMode.KeepAspectRatio)
            self.pip_label.setPixmap(QPixmap.fromImage(qi))

        # FPS
        now = time.time()
        self.fps = self.fps * 0.9 + (1.0 / max(now - self._fps_t, 1e-6)) * 0.1
        self._fps_t = now

        # Status bar
        gi = {"open":"ROT","pinch":"SCL","fist":"---","peace":"RST","point":"PAN","other":"IDL",None:"D/C"}
        gl, gr = self.last_gesture
        self.lbl_gestures.setText(f"L: [{gi.get(gl,'D/C')}]  |  R: [{gi.get(gr,'D/C')}]")
        rm = "X-RAY" if self.current_view.get("xray_mode") else ("WIRE" if self.btn_wire.isChecked() else "SOLID")
        cam = "ON" if self.tracker.camera_ok else "OFF"
        eng = "GL" if self._use_gl else "SW"
        self.status.showMessage(
            f">>> FPS: {self.fps:04.1f}  |  DB: {self.current_name}  |  "
            f"MODE: {rm}  |  CAM: {cam}  |  ENGINE: {eng}")

    def closeEvent(self, e):
        self._timer.stop(); self.tracker.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    p = app.palette(); p.setColor(p.ColorRole.Window, QColor(5, 8, 15))
    app.setPalette(p)
    v = AppViewer(); v.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()