"""
gl_renderer.py — Hardware-Accelerated OpenGL 3D Renderer

QOpenGLWidget-based viewport with multi-light Phong shading, X-ray mode,
wireframe mode, orientation gizmo, and loading spinner.
Requires: PyOpenGL, PyQt6.
"""

import ctypes
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QFont, QColor, QPen
from PyQt6.QtWidgets import QWidget

_GL_OK = False
try:
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL.GL import *
    from OpenGL.GL import shaders as _gl_shaders
    _GL_OK = True
except ImportError:
    # Provide a dummy base so the module can still be imported
    from PyQt6.QtWidgets import QLabel as QOpenGLWidget  # type: ignore

logger = logging.getLogger(__name__)

# ─── Shader Sources ──────────────────────────────────────────────────────────

_VERT_MESH = """
#version 330 core
layout(location = 0) in vec3 aPos;     // Vertex position (model space)
layout(location = 1) in vec3 aNormal;  // Per-vertex smooth normal (model space)

uniform mat4 model;        // Model transform (rotation + scale + pan)
uniform mat4 view;         // View transform (camera look-at)
uniform mat4 projection;   // Perspective projection
uniform mat3 normalMatrix; // Inverse-transpose of model's upper 3×3 (corrects non-uniform scaling)

out vec3 FragPos;   // World-space position passed to fragment shader
out vec3 Normal;    // World-space normal passed to fragment shader

void main() {
    vec4 wp = model * vec4(aPos, 1.0);
    FragPos = wp.xyz;                              // World position for lighting
    Normal  = normalize(normalMatrix * aNormal);   // Transform normal to world space
    gl_Position = projection * view * wp;
}
"""

_FRAG_MESH = """
#version 330 core
in vec3 FragPos;
in vec3 Normal;

// Three-light structure matching the Python LIGHTS constant.
struct Light { vec3 direction; vec3 color; float intensity; };

uniform Light lights[3];
uniform int   numLights;
uniform vec3  objectColor;
uniform vec3  viewPos;      // Camera position in world space (fixed at [0,0,3])
uniform float ambient;      // Minimum illumination (prevents pure-black shadows)
uniform float rimPower;     // Fresnel rim exponent (higher = tighter edge glow)
uniform float rimIntensity; // Fresnel rim strength multiplier
uniform int   renderMode;   // 0=solid  1=xray  2=wireframe
uniform float xrayDensity;  // Controls opacity/saturation of the X-ray effect

out vec4 FragColor;

void main() {
    vec3 N = normalize(Normal);
    vec3 V = normalize(viewPos - FragPos);

    // ── X-Ray Mode (additive alpha blend) ────────────────────────────────
    // Fragments facing the camera (N·V ≈ 1) are nearly transparent;
    // silhouette edges (N·V ≈ 0) are opaque. This simulates volumetric density.
    if (renderMode == 1) {
        float edge = 1.0 - abs(dot(N, V));   // 0 at face-on, 1 at silhouette
        edge = pow(edge, 1.2);               // Non-linear ramp for sharper edges
        float a = xrayDensity * (0.15 + 0.85 * edge);  // Base + edge boost
        // Colour ramp: dark teal at face-on → bright cyan at edges
        vec3 c  = mix(vec3(0.02, 0.15, 0.25), vec3(0.1, 0.85, 1.0), edge);
        FragColor = vec4(c, a);
        return;
    }

    // ── Wireframe Mode ────────────────────────────────────────────────────
    // Depth-based brightness: closer fragments appear brighter.
    // gl_FragCoord.z is NDC depth in [0, 1]; we invert it for a near=bright effect.
    if (renderMode == 2) {
        float d = 1.0 - gl_FragCoord.z;      // Near → 1.0, Far → 0.0
        float b = 0.25 + 0.75 * d;           // Clamp between 25%–80% brightness
        FragColor = vec4(objectColor * b, 1.0);
        return;
    }

    // ── Solid Phong Shading ───────────────────────────────────────────────
    // Ambient: constant base illumination to avoid completely black shadows.
    vec3 result = ambient * objectColor;

    // Diffuse + specular contribution from each light (Blinn-Phong half vector).
    for (int i = 0; i < numLights; i++) {
        vec3  L    = normalize(-lights[i].direction); // Light dir is toward light
        float diff = max(dot(N, L), 0.0);             // Lambertian term
        vec3  H    = normalize(L + V);                // Blinn-Phong half vector
        float spec = pow(max(dot(N, H), 0.0), 32.0);  // Specular highlight
        result += lights[i].intensity * lights[i].color
                  * (diff * objectColor + spec * 0.3); // 30% white spec highlight
    }
    float rim = pow(1.0 - abs(dot(N, V)), rimPower) * rimIntensity;
    result += rim * vec3(0.3, 0.5, 0.8);
    FragColor = vec4(result, 1.0);
}
"""

_VERT_UTIL = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 mvp;
out vec3 vColor;
void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

_FRAG_UTIL = """
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""

# ─── Matrix Helpers ──────────────────────────────────────────────────────────

def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = near - far
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / nf, 2 * far * near / nf],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def _look_at(eye, center, up) -> np.ndarray:
    e, c, u = np.asarray(eye, np.float32), np.asarray(center, np.float32), np.asarray(up, np.float32)
    f = c - e; f /= np.linalg.norm(f)
    s = np.cross(f, u); s /= np.linalg.norm(s)
    u2 = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s; M[1, :3] = u2; M[2, :3] = -f
    T = np.eye(4, dtype=np.float32); T[:3, 3] = -e
    return M @ T


def _rot4(rx_deg: float, ry_deg: float) -> np.ndarray:
    rx, ry = math.radians(rx_deg), math.radians(ry_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
    return Ry @ Rx


def _scale4(s: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float32); M[0,0] = M[1,1] = M[2,2] = s; return M


def _translate4(x: float, y: float, z: float) -> np.ndarray:
    M = np.eye(4, dtype=np.float32); M[0,3] = x; M[1,3] = y; M[2,3] = z; return M

# ─── Light Presets ───────────────────────────────────────────────────────────
LIGHTS = [
    {"dir": np.array([0.4, 0.7, -0.6],  dtype=np.float32), "color": (1.0, 0.98, 0.95), "intensity": 0.6},
    {"dir": np.array([-0.5, 0.3, -0.4], dtype=np.float32), "color": (0.7, 0.85, 1.0),  "intensity": 0.3},
    {"dir": np.array([0.0, -0.5, 0.8],  dtype=np.float32), "color": (1.0, 0.9, 0.8),   "intensity": 0.15},
]


# ═════════════════════════════════════════════════════════════════════════════
if _GL_OK:

    class GLViewport(QOpenGLWidget):
        """Hardware-accelerated 3D mesh viewport using OpenGL 3.3 core profile.

        Inherits from QOpenGLWidget — Qt manages the OpenGL context lifetime.
        The viewport owns all GPU resources (VAOs, VBOs, EBO, shader programs)
        and must be used only from the **main GUI thread**.

        Rendering features:
            - GLSL 330 Phong fragment shader with 3-point lighting + rim glow.
            - X-ray mode: additive alpha blend (GL_SRC_ALPHA, GL_ONE).
            - Wireframe mode: GL_LINE polygon mode.
            - Orientation gizmo: 3 colored axis lines in a sub-viewport
              (bottom-right corner) with QPainter text label overlay.
            - Loading spinner: rotating arc drawn as a GL_LINE_STRIP.
            - FPS counter: rendered via QPainter after native GL drawing.

        GPU resource lifecycle:
            ``initializeGL()``   — called once by Qt after context creation.
                                    Compiles shaders, allocates gizmo + spinner VAOs.
            ``set_mesh()``       — called from the GUI thread after a mesh is loaded.
                                    Uploads vertex + index data to the GPU.
            ``paintGL()``        — called by Qt on each repaint (triggered by update()).
            Window close         — Qt destroys the context; no explicit cleanup needed.

        Thread safety:
            All public methods (set_mesh, set_view, set_color, etc.) must be called
            from the **main thread**. ``set_mesh()`` calls ``makeCurrent()`` /
            ``doneCurrent()`` internally to satisfy OpenGL context requirements.

        Communication pattern:
            AppViewer holds a reference (``self.viewport``) and calls the set_*
            methods each tick from ``_tick()``. GLViewport does NOT call back into
            AppViewer except through the ``app_ref`` attribute for mouse events.

        Args:
            parent: Optional parent QWidget.

        Attributes:
            app_ref: Back-reference to :class:`AppViewer` set after construction.
                     Used by mouse event handlers to mutate ``target_view``.
        """

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self.setMouseTracking(True)
            self.app_ref = None

            # View state (driven by AppViewer)
            self._rx = 20.0; self._ry = -30.0
            self._zoom = 1.0; self._px = 0.0; self._py = 0.0
            self._xray = False; self._wireframe = False
            self._xray_density = 0.3
            self._color = (0.85, 0.85, 0.85)
            self._loading = False
            self._load_t0 = 0.0
            self._bg = (0.02, 0.03, 0.06)
            self._fps = 0.0

            # GL objects
            self._gl_ready = False
            self._mesh_vao = None; self._mesh_vbo = None; self._mesh_ebo = None
            self._face_count = 0; self._has_mesh = False
            self._gizmo_vao = None; self._gizmo_vbo = None
            self._spinner_vao = None; self._spinner_vbo = None
            self._prog_mesh = None; self._prog_util = None
            self._gizmo_label_pos: List[Tuple[float, float]] = []

            # Mouse
            self._m_down = False; self._m_btn = None; self._mx = 0; self._my = 0

        # ── GL Lifecycle ──────────────────────────────────────────────────
        def initializeGL(self) -> None:
            try:
                glClearColor(*self._bg, 1.0)
                glEnable(GL_DEPTH_TEST)
                glEnable(GL_MULTISAMPLE)
                glLineWidth(2.0)

                # Compile shaders
                self._prog_mesh = _gl_shaders.compileProgram(
                    _gl_shaders.compileShader(_VERT_MESH, GL_VERTEX_SHADER),
                    _gl_shaders.compileShader(_FRAG_MESH, GL_FRAGMENT_SHADER))
                self._prog_util = _gl_shaders.compileProgram(
                    _gl_shaders.compileShader(_VERT_UTIL, GL_VERTEX_SHADER),
                    _gl_shaders.compileShader(_FRAG_UTIL, GL_FRAGMENT_SHADER))

                # Gizmo geometry (3 axis lines × 2 verts, each with pos+color)
                gdata = np.array([
                    0,0,0, 1,0.2,0.2,  0.9,0,0, 1,0.2,0.2,
                    0,0,0, 0.2,1,0.2,  0,0.9,0, 0.2,1,0.2,
                    0,0,0, 0.2,0.4,1,  0,0,0.9, 0.2,0.4,1,
                ], dtype=np.float32)
                self._gizmo_vao = glGenVertexArrays(1)
                self._gizmo_vbo = glGenBuffers(1)
                glBindVertexArray(self._gizmo_vao)
                glBindBuffer(GL_ARRAY_BUFFER, self._gizmo_vbo)
                glBufferData(GL_ARRAY_BUFFER, gdata.nbytes, gdata, GL_STATIC_DRAW)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
                glEnableVertexAttribArray(1)
                glBindVertexArray(0)

                # Spinner arc geometry
                N = 50; arc_frac = 0.75
                sdata = []
                for i in range(int(N * arc_frac) + 1):
                    a = (i / N) * 2 * math.pi
                    sdata.extend([math.cos(a)*0.08, math.sin(a)*0.08, 0, 0, 0.9, 1])
                sdata = np.array(sdata, dtype=np.float32)
                self._spinner_vao = glGenVertexArrays(1)
                self._spinner_vbo = glGenBuffers(1)
                self._spinner_n = int(N * arc_frac) + 1
                glBindVertexArray(self._spinner_vao)
                glBindBuffer(GL_ARRAY_BUFFER, self._spinner_vbo)
                glBufferData(GL_ARRAY_BUFFER, sdata.nbytes, sdata, GL_STATIC_DRAW)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
                glEnableVertexAttribArray(1)
                glBindVertexArray(0)

                self._gl_ready = True
                logger.info("OpenGL initialized: %s", glGetString(GL_VERSION))
            except Exception as e:
                logger.error("GL init failed: %s", e)
                self._gl_ready = False

        def resizeGL(self, w: int, h: int) -> None:
            if self._gl_ready:
                glViewport(0, 0, w, h)

        def paintGL(self) -> None:
            if not self._gl_ready:
                return
            painter = QPainter(self)
            painter.beginNativePainting()

            glClearColor(*self._bg, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            w, h = self.width(), self.height()
            aspect = w / max(h, 1)
            proj = _perspective(60.0, aspect, 0.01, 100.0)
            view = _look_at([0, 0, 3], [0, 0, 0], [0, 1, 0])

            if self._has_mesh and not self._loading:
                self._draw_mesh(proj, view, w, h)
            if self._loading:
                self._draw_spinner_gl(proj)

            # Gizmo (always visible when not loading)
            if not self._loading:
                self._draw_gizmo(proj, w, h)

            painter.endNativePainting()

            # QPainter 2D overlays
            if self._loading:
                self._draw_loading_text(painter, w, h)
            if not self._loading:
                self._draw_gizmo_labels(painter)

            # FPS at bottom-left
            painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            painter.setPen(QColor(0, 200, 255, 180))
            painter.drawText(12, h - 10, f"FPS: {self._fps:.1f}")

            painter.end()

        # ── Mesh Upload ───────────────────────────────────────────────────
        def set_mesh(self, mesh: dict) -> None:
            """Upload mesh data to the GPU."""
            self.makeCurrent()
            verts = mesh["verts"]
            faces = mesh["faces"]
            vnorms = mesh.get("vertex_normals")
            if vnorms is None:
                vnorms = np.zeros_like(verts)

            vdata = np.hstack([verts, vnorms]).astype(np.float32)
            fdata = faces.astype(np.uint32)

            if self._mesh_vao is None:
                self._mesh_vao = glGenVertexArrays(1)
                self._mesh_vbo = glGenBuffers(1)
                self._mesh_ebo = glGenBuffers(1)

            glBindVertexArray(self._mesh_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._mesh_vbo)
            glBufferData(GL_ARRAY_BUFFER, vdata.nbytes, vdata, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._mesh_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, fdata.nbytes, fdata, GL_STATIC_DRAW)
            glBindVertexArray(0)

            self._face_count = len(faces) * 3
            self._has_mesh = True
            self.doneCurrent()
            logger.info("Mesh uploaded to GPU: %d faces", len(faces))

        def set_view(self, v: dict) -> None:
            """Update view parameters from the application state."""
            self._rx = float(v.get("rot_x", self._rx))
            self._ry = float(v.get("rot_y", self._ry))
            self._zoom = float(v.get("zoom", self._zoom))
            self._px = float(v.get("pan_x", self._px))
            self._py = float(v.get("pan_y", self._py))
            self._xray = bool(v.get("xray_mode", False))
            self._xray_density = float(v.get("xray_density", 0.3))

        def set_color(self, c: Tuple[float, float, float]) -> None:
            self._color = c

        def set_wireframe(self, on: bool) -> None:
            self._wireframe = on

        def set_loading(self, on: bool) -> None:
            if on and not self._loading:
                self._load_t0 = time.time()
            self._loading = on

        def set_fps(self, fps: float) -> None:
            self._fps = fps

        def set_bg(self, r: float, g: float, b: float) -> None:
            self._bg = (r, g, b)

        def grab_screenshot(self):
            """Return a QImage of the current viewport."""
            return self.grabFramebuffer()

        # ── Internal Rendering ────────────────────────────────────────────
        def _draw_mesh(self, proj, view_mat, w, h):
            model = (_translate4(self._px * 0.003, -self._py * 0.003, 0)
                     @ _scale4(self._zoom)
                     @ _rot4(self._rx, self._ry))
            nm3 = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

            glUseProgram(self._prog_mesh)
            # Matrices
            glUniformMatrix4fv(glGetUniformLocation(self._prog_mesh, "model"), 1, GL_TRUE, model)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_mesh, "view"), 1, GL_TRUE, view_mat)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_mesh, "projection"), 1, GL_TRUE, proj)
            glUniformMatrix3fv(glGetUniformLocation(self._prog_mesh, "normalMatrix"), 1, GL_TRUE, nm3)
            # Material
            glUniform3f(glGetUniformLocation(self._prog_mesh, "objectColor"), *self._color)
            glUniform3f(glGetUniformLocation(self._prog_mesh, "viewPos"), 0, 0, 3)
            glUniform1f(glGetUniformLocation(self._prog_mesh, "ambient"), 0.22)
            glUniform1f(glGetUniformLocation(self._prog_mesh, "rimPower"), 2.5)
            glUniform1f(glGetUniformLocation(self._prog_mesh, "rimIntensity"), 0.55)
            glUniform1f(glGetUniformLocation(self._prog_mesh, "xrayDensity"), self._xray_density)

            # Lights
            glUniform1i(glGetUniformLocation(self._prog_mesh, "numLights"), len(LIGHTS))
            for i, lt in enumerate(LIGHTS):
                d = lt["dir"] / np.linalg.norm(lt["dir"])
                glUniform3f(glGetUniformLocation(self._prog_mesh, f"lights[{i}].direction"), *d)
                glUniform3f(glGetUniformLocation(self._prog_mesh, f"lights[{i}].color"), *lt["color"])
                glUniform1f(glGetUniformLocation(self._prog_mesh, f"lights[{i}].intensity"), lt["intensity"])

            # Render mode
            if self._xray:
                glUniform1i(glGetUniformLocation(self._prog_mesh, "renderMode"), 1)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glDepthMask(GL_FALSE)
                glDisable(GL_CULL_FACE)
            elif self._wireframe:
                glUniform1i(glGetUniformLocation(self._prog_mesh, "renderMode"), 2)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glUniform1i(glGetUniformLocation(self._prog_mesh, "renderMode"), 0)
                glEnable(GL_DEPTH_TEST)
                glDepthMask(GL_TRUE)

            glBindVertexArray(self._mesh_vao)
            glDrawElements(GL_TRIANGLES, self._face_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

            # Restore state
            if self._xray:
                glDisable(GL_BLEND)
                glDepthMask(GL_TRUE)
            if self._wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        def _draw_gizmo(self, proj, w, h):
            gs = min(120, min(w, h) // 5)
            gx = w - gs - 10
            gy = 10
            glViewport(gx, gy, gs, gs)
            glClear(GL_DEPTH_BUFFER_BIT)

            gp = _perspective(45.0, 1.0, 0.1, 100.0)
            gv = _look_at([0, 0, 3], [0, 0, 0], [0, 1, 0])
            gm = _rot4(self._rx, self._ry)
            mvp = gp @ gv @ gm

            glUseProgram(self._prog_util)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_util, "mvp"), 1, GL_TRUE, mvp)
            glBindVertexArray(self._gizmo_vao)
            glDrawArrays(GL_LINES, 0, 6)
            glBindVertexArray(0)

            glViewport(0, 0, w, h)

            # Compute label screen positions
            self._gizmo_label_pos = []
            tips = np.array([[0.9,0,0,1],[0,0.9,0,1],[0,0,0.9,1]], dtype=np.float32)
            for tip in tips:
                clip = mvp @ tip
                if abs(clip[3]) < 1e-6:
                    self._gizmo_label_pos.append((gx + gs//2, h - gy - gs//2))
                    continue
                ndc = clip[:3] / clip[3]
                sx = gx + (ndc[0] + 1) / 2 * gs
                sy = h - (gy + (ndc[1] + 1) / 2 * gs)
                self._gizmo_label_pos.append((sx, sy))

        def _draw_gizmo_labels(self, painter: QPainter):
            labels = [("X", QColor(255, 80, 80)), ("Y", QColor(80, 255, 80)), ("Z", QColor(100, 130, 255))]
            painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            for (lbl, col), (sx, sy) in zip(labels, self._gizmo_label_pos):
                painter.setPen(col)
                painter.drawText(int(sx) + 4, int(sy) - 2, lbl)

        def _draw_spinner_gl(self, proj):
            elapsed = time.time() - self._load_t0
            angle = elapsed * 200
            rot_z = np.eye(4, dtype=np.float32)
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            rot_z[0,0] = c; rot_z[0,1] = -s; rot_z[1,0] = s; rot_z[1,1] = c
            ortho = np.eye(4, dtype=np.float32)

            glUseProgram(self._prog_util)
            glUniformMatrix4fv(glGetUniformLocation(self._prog_util, "mvp"), 1, GL_TRUE, ortho @ rot_z)
            glLineWidth(3.0)
            glBindVertexArray(self._spinner_vao)
            glDrawArrays(GL_LINE_STRIP, 0, self._spinner_n)
            glBindVertexArray(0)
            glLineWidth(2.0)

        def _draw_loading_text(self, painter: QPainter, w: int, h: int):
            painter.setFont(QFont("Consolas", 12))
            painter.setPen(QColor(0, 140, 220))
            dots = "." * (int(time.time() * 3) % 4)
            painter.drawText(w // 2 - 110, h // 2 + 75, f"LOADING DATABANK{dots}")

        # ── Mouse Events ──────────────────────────────────────────────────
        def mousePressEvent(self, ev):
            b = ev.button()
            if b == Qt.MouseButton.LeftButton:
                self._m_btn = "L"
            elif b == Qt.MouseButton.RightButton:
                self._m_btn = "R"
            else:
                return
            self._m_down = True; self._mx = ev.pos().x(); self._my = ev.pos().y()

        def mouseReleaseEvent(self, ev):
            self._m_down = False
            if self.app_ref:
                self.app_ref.interaction_points.clear()

        def mouseMoveEvent(self, ev):
            if not self._m_down or not self.app_ref:
                return
            x, y = ev.pos().x(), ev.pos().y()
            dx, dy = x - self._mx, y - self._my
            if self._m_btn == "L":
                self.app_ref.target_view["rot_y"] += dx * 0.4
                self.app_ref.target_view["rot_x"] += dy * 0.4
            elif self._m_btn == "R":
                self.app_ref.target_view["pan_x"] += dx * 0.6
                self.app_ref.target_view["pan_y"] += dy * 0.6
            self._mx, self._my = x, y

        def wheelEvent(self, ev):
            if not self.app_ref:
                return
            if ev.angleDelta().y() > 0:
                self.app_ref.target_view["zoom"] *= 1.1
            else:
                self.app_ref.target_view["zoom"] /= 1.1
            self.app_ref.target_view["zoom"] = max(0.1, min(10.0, self.app_ref.target_view["zoom"]))
