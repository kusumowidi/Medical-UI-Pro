"""
renderer.py — Enhanced Software 3D Renderer (CPU fallback)

Provides a pure-CPU rasterizer used when OpenGL / PyOpenGL is unavailable.
All rendering is done via NumPy array operations and OpenCV drawing primitives.

Rendering pipeline
------------------
1. Apply rotation (rot_y ◦ rot_x) and zoom/pan transforms in world space.
2. Perspective-project 3D vertices to 2D pixel coordinates.
3. Dispatch to the appropriate rendering mode:
   a. *X-ray* -- additive point-density accumulation with Gaussian blur + Canny edges.
   b. *Point cloud* -- depth-shaded vertex dots (auto for meshes > BATCH_FACE_LIMIT).
   c. *Solid / Wireframe / Mixed* -- back-face culled, painter’s-algorithm sorted triangles
      with multi-light Phong + rim shading via cv2.fillPoly / cv2.polylines.
4. Draw the orientation gizmo (XYZ arrows, bottom-right corner).

Style field contract
--------------------
This renderer branches on ``mesh["style"]``:

- ``"solid"``      — filled Phong-lit triangles.
- ``"wireframe"``  — triangle edge lines only.
- ``"mixed"``      — filled triangles + wireframe overlay.
- ``"pointcloud"`` — vertex dot cloud (also triggered when face count > BATCH_FACE_LIMIT).

See models.py for the full Mesh Dict Contract.
"""


import logging
import math
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
BATCH_FACE_LIMIT: int = 15_000
BG_COLOR_DARK: Tuple[int, int, int] = (15, 10, 5)
BG_COLOR_LIGHT: Tuple[int, int, int] = (235, 235, 240)
BACKFACE_THRESHOLD: float = 0.15
CAMERA_OFFSET_Z: float = 3.0
PAN_SCALE_FACTOR: float = 0.003

# 3-point lighting (#6)
LIGHTS = [
    {"dir": np.array([0.4, 0.7, -0.6], np.float32), "color": np.array([1.0, 0.98, 0.95]), "i": 0.6},
    {"dir": np.array([-0.5, 0.3, -0.4], np.float32), "color": np.array([0.7, 0.85, 1.0]),  "i": 0.3},
    {"dir": np.array([0.0, -0.5, 0.8], np.float32),  "color": np.array([1.0, 0.9, 0.8]),   "i": 0.15},
]
VIEW_DIR = np.array([0.0, 0.0, -1.0], np.float32)

# Normalize light directions at module load
for lt in LIGHTS:
    lt["dir"] = lt["dir"] / np.linalg.norm(lt["dir"])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)


class SoftwareRenderer:
    """CPU-based 3D renderer with multi-light Phong shading.

    Designed as a fallback for systems where PyOpenGL is unavailable or fails
    to initialize. It renders into a pre-allocated NumPy (H, W, 3) BGR canvas
    which is then converted to a QImage and shown in an InteractiveCanvas.

    Performance characteristics:
        - Solid/wireframe mode scales with face count; performance degrades
          steeply above BATCH_FACE_LIMIT (auto-falls back to point cloud).
        - X-ray mode scales with vertex count and uses Gaussian blur (O(n)²).
        - Point-cloud mode is O(vertices) and GPU-friendly via NumPy scatter.

    Thread safety:
        Not thread-safe. All render() calls must originate from the GUI thread.
        Reconstruction of the renderer (new SoftwareRenderer()) is safe to call
        from any thread before passing to the GUI thread.

    Args:
        width:  Canvas width in pixels.
        height: Canvas height in pixels.
        fov:    Vertical field-of-view in degrees (default 60°).
        dark:   If True, use dark background (radiology dark mode);
                if False, use light background (OR room mode).
    """

    def __init__(self, width: int, height: int, fov: float = 60.0, dark: bool = True) -> None:
        self.w = width
        self.h = height
        self.fov = fov
        self._dark = dark
        bg = BG_COLOR_DARK if dark else BG_COLOR_LIGHT
        self.bg = np.full((self.h, self.w, 3), bg, dtype=np.uint8)

    def project(self, v3d: np.ndarray):
        f = self.h / (2.0 * math.tan(math.radians(self.fov / 2.0)))
        cx, cy = self.w / 2.0, self.h / 2.0
        z = v3d[:, 2] + 1e-6
        px = v3d[:, 0] * f / z + cx
        py = -v3d[:, 1] * f / z + cy
        return np.stack([px, py], axis=1), v3d[:, 2]

    def compute_normals(self, v, faces):
        n = np.cross(v[faces[:,1]] - v[faces[:,0]], v[faces[:,2]] - v[faces[:,0]])
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

    def render(self, canvas: np.ndarray, mesh: dict, view: dict) -> None:
        verts = mesh["verts"].copy()
        faces = mesh["faces"]
        color = mesh.get("color", (0.7, 0.4, 0.3))
        is_xray = view.get("xray_mode", False)
        np.copyto(canvas, self.bg)
        if len(verts) == 0:
            return

        R = rot_y(math.radians(view["rot_y"])) @ rot_x(math.radians(view["rot_x"]))
        ps = 1.5 / (view["zoom"] + 1e-6)
        verts = verts @ R.T
        verts[:, 0] += view["pan_x"] * ps * PAN_SCALE_FACTOR
        verts[:, 1] -= view["pan_y"] * ps * PAN_SCALE_FACTOR
        verts *= view["zoom"]
        verts[:, 2] += CAMERA_OFFSET_Z
        px2d, pz = self.project(verts)

        if is_xray:
            self._xray(canvas, px2d, pz, faces, view)
            return

        style = mesh.get("style", "solid")
        if len(faces) > BATCH_FACE_LIMIT or style == "pointcloud":
            self._pointcloud(canvas, px2d, pz, color)
            return

        # Multi-light Phong (#6)
        normals = mesh["normals"] @ R.T if "normals" in mesh else self.compute_normals(verts, faces)

        intensity = np.full(len(normals), 0.22, dtype=np.float32)
        for lt in LIGHTS:
            diff = np.clip(np.dot(normals, lt["dir"]), 0, 1)
            intensity += diff * lt["i"]
        rim = 1.0 - np.abs(np.dot(normals, VIEW_DIR))
        intensity += np.clip(np.power(rim, 2.5), 0, 1) * 0.55

        face_z = pz[faces].mean(axis=1)
        facing = np.dot(normals, VIEW_DIR) < BACKFACE_THRESHOLD
        order = np.argsort(-face_z)
        r0, g0, b0 = color
        px_int = px2d.astype(np.int32)

        for fi in order:
            if facing[fi]:
                continue
            pts = px_int[faces[fi]]
            if pts[:,0].max() < 0 or pts[:,0].min() >= self.w or pts[:,1].max() < 0 or pts[:,1].min() >= self.h:
                continue
            lit = float(intensity[fi])
            r, g, b = int(min(255, r0*255*lit)), int(min(255, g0*255*lit)), int(min(255, b0*255*lit))
            if style in ("solid", "mixed"):
                cv2.fillPoly(canvas, [pts], (b, g, r))
            if style in ("wireframe", "mixed"):
                eb = int(lit * 100 + 40)
                cv2.polylines(canvas, [pts], True, (eb, eb+30, eb+50), 1, cv2.LINE_AA)

        # Orientation gizmo (#10)
        self._draw_gizmo(canvas, R)

    # ── Depth-shaded point cloud (#2) ─────────────────────────────────
    def _pointcloud(self, canvas, px2d, pz, color):
        valid = (pz > 0.1) & (px2d[:,0] >= 0) & (px2d[:,0] < self.w) & (px2d[:,1] >= 0) & (px2d[:,1] < self.h)
        pts = px2d[valid].astype(np.int32)
        pz_v = pz[valid]
        if len(pts) == 0:
            return
        z_min, z_max = pz_v.min(), pz_v.max()
        z_norm = 1.0 - np.clip((pz_v - z_min) / (z_max - z_min + 1e-6), 0, 1)
        r0, g0, b0 = (int(c * 255) for c in color)
        intensities = 0.3 + 0.7 * z_norm
        colors_b = np.clip(b0 * intensities, 0, 255).astype(np.uint8)
        colors_g = np.clip(g0 * intensities, 0, 255).astype(np.uint8)
        colors_r = np.clip(r0 * intensities, 0, 255).astype(np.uint8)
        bgr = np.stack([colors_b, colors_g, colors_r], axis=-1)
        canvas[pts[:,1], pts[:,0]] = bgr

    # ── X-Ray with edge contours ──────────────────────────────────────
    def _xray(self, canvas, px2d, pz, faces, view):
        """Render an X-ray visualization via additive point-density accumulation.

        Algorithm:
            1. Scatter visible vertices into a float32 mask weighted by depth.
            2. Blend the raw mask with a Gaussian-blurred version for a soft glow.
            3. Map to a cyan colour ramp and additively blend into the canvas.
            4. Run Canny edge detection on the mask and overlay in cyan.

        Args:
            canvas: (H, W, 3) uint8 BGR image to draw into.
            px2d:   (N, 2) projected 2D pixel positions.
            pz:     (N,) depth values in camera space.
            faces:  (M, 3) triangle indices (unused — vertex density only).
            view:   View state dict; uses ``xray_density`` key.
        """
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        valid = (
            (pz > 0.1)
            & (px2d[:, 0] >= 0) & (px2d[:, 0] < self.w)
            & (px2d[:, 1] >= 0) & (px2d[:, 1] < self.h)
        )
        pts = px2d[valid].astype(np.int32)
        pz_v = pz[valid]
        if len(pts) == 0:
            return
        z_min, z_max = pz_v.min(), pz_v.max()
        # Depth weight: nearer vertices contribute more density
        dw = 1.0 - np.clip((pz_v - z_min) / (z_max - z_min + 1e-6), 0, 1) ** 2
        density = view.get("xray_density", 0.3)
        np.add.at(mask, (pts[:, 1], pts[:, 0]), dw * density)

        ms = mask.copy()
        mb = cv2.GaussianBlur(mask, (5, 5), 1.0)
        # Guard: GaussianBlur may squeeze shape on degenerate input
        if mb.shape != mask.shape:
            mb = np.zeros_like(mask)
        mf = ms * 0.8 + mb * 1.5
        mx = np.percentile(mf, 99.9) + 1e-6
        mf = np.clip(mf / mx, 0, 1)
        # Ensure mf is always (H, W) regardless of any NumPy/OpenCV shape edge cases
        mf = np.reshape(mf, (self.h, self.w))

        # Cyan colour ramp overlay (additive blend)
        xc = np.zeros((self.h, self.w, 3), dtype=np.float32)
        xc[:, :, 0] = mf * 255
        xc[:, :, 1] = mf * np.power(mf, 0.4) * 230
        xc[:, :, 2] = mf * np.power(mf, 1.5) * 110
        # Use explicit numpy addition + clip to avoid cv2.add shape quirks
        canvas[:] = np.clip(canvas.astype(np.float32) + xc, 0, 255).astype(np.uint8)

        # Edge contour overlay via Canny for anatomical boundary clarity
        canny_input = (mf * 255).astype(np.uint8).reshape(self.h, self.w)
        edges = cv2.Canny(canny_input, 30, 100)
        # Guard against empty/squeezed return from Canny on degenerate input
        if edges.shape != (self.h, self.w):
            edges = np.zeros((self.h, self.w), dtype=np.uint8)
        edge_color = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        edge_color[:, :, 0] = edges
        edge_color[:, :, 1] = (edges * 0.6).astype(np.uint8)
        canvas[:] = np.clip(canvas.astype(np.float32) + edge_color.astype(np.float32), 0, 255).astype(np.uint8)

    # ── Orientation Gizmo ─────────────────────────────────────────────
    def _draw_gizmo(self, canvas, R):
        size = 45
        ox, oy = self.w - size - 20, self.h - size - 20
        center = np.array([ox, oy])
        axes = np.eye(3, dtype=np.float32) @ R.T
        colors = [(0, 0, 255), (0, 220, 0), (255, 100, 60)]
        labels = ["X", "Y", "Z"]
        for i in range(3):
            end = center + np.array([axes[i,0], -axes[i,1]]) * (size * 0.8)
            cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(end.astype(int)), colors[i], 2, tipLength=0.25)
            tx, ty = int(end[0]) + 5, int(end[1]) + 5
            cv2.putText(canvas, labels[i], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1, cv2.LINE_AA)
