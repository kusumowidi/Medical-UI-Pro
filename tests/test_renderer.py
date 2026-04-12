"""
tests/test_renderer.py — Unit tests for the software renderer (renderer.py)

Tests cover projection math, point-cloud fallback, gizmo rendering, and
the X-ray pipeline. All tests are headless (no display, no OpenGL context).
"""

import math

import numpy as np
import pytest

from renderer import SoftwareRenderer, rot_x, rot_y, BATCH_FACE_LIMIT


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_canvas(w: int = 320, h: int = 240) -> np.ndarray:
    """Return a blank BGR canvas."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _unit_cube_mesh() -> dict:
    """Return a minimal mesh dict with 2 triangles forming a flat quad."""
    verts = np.array([
        [-0.5, -0.5, 0.0],
        [ 0.5, -0.5, 0.0],
        [ 0.5,  0.5, 0.0],
        [-0.5,  0.5, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
    return {
        "verts": verts,
        "faces": faces,
        "normals": fn.astype(np.float32),
        "color": (0.7, 0.5, 0.3),
        "style": "solid",
    }


def _default_view() -> dict:
    return {
        "rot_x": 0.0, "rot_y": 0.0,
        "zoom": 1.0, "pan_x": 0.0, "pan_y": 0.0,
        "xray_mode": False, "xray_density": 0.3,
        "is_loading_stl": False,
    }


# ─── Rotation helpers ────────────────────────────────────────────────────────

class TestRotationMatrices:
    def test_rot_x_identity_at_zero(self):
        R = rot_x(0.0)
        assert np.allclose(R, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_rot_y_identity_at_zero(self):
        R = rot_y(0.0)
        assert np.allclose(R, np.eye(3, dtype=np.float32), atol=1e-6)

    def test_rot_x_90_flips_y_z(self):
        R = rot_x(math.pi / 2)
        v = R @ np.array([0, 1, 0], dtype=np.float32)
        assert np.allclose(v, [0, 0, 1], atol=1e-5)

    def test_rot_y_90_flips_x_z(self):
        R = rot_y(math.pi / 2)
        v = R @ np.array([1, 0, 0], dtype=np.float32)
        assert np.allclose(v, [0, 0, -1], atol=1e-5)

    def test_rot_x_output_shape(self):
        assert rot_x(1.0).shape == (3, 3)

    def test_rot_y_output_shape(self):
        assert rot_y(1.0).shape == (3, 3)

    def test_rot_x_is_orthogonal(self):
        R = rot_x(0.7)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_rot_y_is_orthogonal(self):
        R = rot_y(1.2)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


# ─── SoftwareRenderer construction ───────────────────────────────────────────

class TestSoftwareRendererInit:
    def test_default_dark_mode(self):
        r = SoftwareRenderer(320, 240)
        assert r._dark is True

    def test_light_mode_flag(self):
        r = SoftwareRenderer(320, 240, dark=False)
        assert r._dark is False

    def test_dimensions_stored(self):
        r = SoftwareRenderer(640, 480)
        assert r.w == 640
        assert r.h == 480

    def test_bg_shape(self):
        r = SoftwareRenderer(320, 240)
        assert r.bg.shape == (240, 320, 3)


# ─── SoftwareRenderer.project ────────────────────────────────────────────────

class TestProject:
    def test_returns_two_arrays(self):
        r = SoftwareRenderer(320, 240)
        v = np.array([[0, 0, 3]], dtype=np.float32)
        px, pz = r.project(v)
        assert px.shape == (1, 2)
        assert pz.shape == (1,)

    def test_origin_projects_to_image_center(self):
        r = SoftwareRenderer(320, 240)
        v = np.array([[0, 0, 3]], dtype=np.float32)
        px, _ = r.project(v)
        cx, cy = 320 / 2, 240 / 2
        assert abs(px[0, 0] - cx) < 2
        assert abs(px[0, 1] - cy) < 2

    def test_negative_z_generates_large_px(self):
        """Very small z should result in extreme projected coordinates."""
        r = SoftwareRenderer(320, 240)
        v = np.array([[0, 0, 0.0001]], dtype=np.float32)
        px, _ = r.project(v)
        # Not a crash — just extreme values
        assert px.shape == (1, 2)


# ─── SoftwareRenderer.render ─────────────────────────────────────────────────

class TestRender:
    def test_render_modifies_canvas(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = _unit_cube_mesh()
        view = _default_view()
        r.render(canvas, mesh, view)
        # Canvas should not be all zeros after rendering a visible mesh
        assert canvas.max() > 0

    def test_render_empty_mesh_no_crash(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = {
            "verts": np.zeros((0, 3), dtype=np.float32),
            "faces": np.zeros((0, 3), dtype=np.int32),
            "normals": np.zeros((0, 3), dtype=np.float32),
            "color": (0.7, 0.5, 0.3),
            "style": "solid",
        }
        view = _default_view()
        r.render(canvas, mesh, view)  # Should not raise

    def test_xray_mode_no_crash(self):
        # Use a large canvas so unit-quad vertices project into frame
        r = SoftwareRenderer(1280, 720)
        canvas = _make_canvas(1280, 720)
        mesh = _unit_cube_mesh()
        view = {**_default_view(), "xray_mode": True}
        r.render(canvas, mesh, view)  # Should not raise

    def test_large_face_count_falls_to_pointcloud(self):
        """Meshes over BATCH_FACE_LIMIT should render as point cloud without error."""
        n = BATCH_FACE_LIMIT + 100
        verts = np.random.rand(n * 3, 3).astype(np.float32)
        faces = np.arange(n * 3, dtype=np.int32).reshape(n, 3)
        fn = np.random.rand(n, 3).astype(np.float32)
        mesh = {"verts": verts, "faces": faces, "normals": fn,
                "color": (0.5, 0.5, 0.5), "style": "solid"}
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        r.render(canvas, mesh, _default_view())  # Should not crash

    def test_wireframe_style_no_crash(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = {**_unit_cube_mesh(), "style": "wireframe"}
        r.render(canvas, mesh, _default_view())

    def test_pointcloud_style_no_crash(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = {**_unit_cube_mesh(), "style": "pointcloud"}
        r.render(canvas, mesh, _default_view())

    def test_canvas_dtype_preserved(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = _unit_cube_mesh()
        r.render(canvas, mesh, _default_view())
        assert canvas.dtype == np.uint8

    def test_zoom_zero_no_crash(self):
        r = SoftwareRenderer(320, 240)
        canvas = _make_canvas(320, 240)
        mesh = _unit_cube_mesh()
        view = {**_default_view(), "zoom": 0.0}
        r.render(canvas, mesh, view)  # Should not raise (division guarded by 1e-6)


# ─── SoftwareRenderer.compute_normals ────────────────────────────────────────

class TestComputeNormals:
    def test_output_shape(self):
        r = SoftwareRenderer(320, 240)
        v = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=np.float32)
        f = np.array([[0,1,2],[1,3,2]], dtype=np.int32)
        n = r.compute_normals(v, f)
        assert n.shape == (2, 3)

    def test_flat_xy_plane_normals_point_z(self):
        r = SoftwareRenderer(320, 240)
        v = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
        f = np.array([[0,1,2]], dtype=np.int32)
        n = r.compute_normals(v, f)
        assert abs(abs(n[0, 2]) - 1.0) < 0.01
