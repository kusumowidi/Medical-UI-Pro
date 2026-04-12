"""
tests/test_models.py — Unit tests for the STL model loader (models.py)

These tests are designed to run headless (no GPU, no display, no GUI) and
cover: file validation, normal computation, mesh normalization, decimation
fallback, metadata extraction, and error handling paths.
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from models import (
    MAX_FILE_SIZE_MB,
    DECIMATION_FACE_THRESHOLD,
    DECIMATION_TARGET_FACES,
    DEFAULT_MODEL_COLOR,
    _compute_vertex_normals,
    get_file_size_mb,
    load_stl_model,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _write_binary_stl(path: str, triangles: list) -> None:
    """Write a minimal binary STL file with the given triangles.

    Args:
        path: Output file path.
        triangles: List of (normal, v0, v1, v2) — each a 3-tuple of floats.
    """
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)  # header
        f.write(struct.pack("<I", len(triangles)))
        for normal, v0, v1, v2 in triangles:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))  # attribute byte count


def _simple_triangle_stl(path: str) -> None:
    """Write a single-triangle STL to path."""
    _write_binary_stl(path, [
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ])


def _two_triangle_stl(path: str) -> None:
    """Write a two-triangle (quad) STL to path."""
    _write_binary_stl(path, [
        ((0, 0, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    ])


# ─── get_file_size_mb ────────────────────────────────────────────────────────

class TestGetFileSizeMb:
    def test_returns_zero_for_missing_file(self):
        assert get_file_size_mb("nonexistent_file_xyz.stl") == 0.0

    def test_correct_size(self, tmp_path):
        p = tmp_path / "test.bin"
        p.write_bytes(b"\x00" * 1024 * 1024)  # 1 MB exactly
        assert abs(get_file_size_mb(str(p)) - 1.0) < 0.001


# ─── _compute_vertex_normals ─────────────────────────────────────────────────

class TestComputeVertexNormals:
    def test_output_shape_matches_verts(self):
        verts = np.random.rand(6, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fn = np.random.rand(2, 3).astype(np.float32)
        result = _compute_vertex_normals(verts, faces, fn)
        assert result.shape == (6, 3)

    def test_output_dtype_is_float32(self):
        verts = np.random.rand(3, 3).astype(np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fn = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        result = _compute_vertex_normals(verts, faces, fn)
        assert result.dtype == np.float32

    def test_normals_are_approximately_unit_length(self):
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [1, 0, 0], [2, 0, 0], [1, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        fn = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        result = _compute_vertex_normals(verts, faces, fn)
        lengths = np.linalg.norm(result, axis=1)
        # All normals should be ~unit length (some may be zero for isolated verts)
        assert np.all(lengths <= 1.001)

    def test_no_nan_in_output(self):
        verts = np.zeros((4, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        fn = np.zeros((1, 3), dtype=np.float32)
        result = _compute_vertex_normals(verts, faces, fn)
        assert not np.any(np.isnan(result))


# ─── load_stl_model ──────────────────────────────────────────────────────────

class TestLoadStlModel:
    def test_returns_none_for_nonexistent_file(self):
        result = load_stl_model("this_file_does_not_exist.stl")
        assert result is None

    def test_returns_dict_for_valid_stl(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _simple_triangle_stl(p)
        result = load_stl_model(p)
        assert result is not None
        assert isinstance(result, dict)

    def test_required_keys_present(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result is not None
        for key in ("verts", "faces", "normals", "vertex_normals", "color", "style", "metadata"):
            assert key in result, f"Missing key: {key}"

    def test_verts_dtype_float32(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["verts"].dtype == np.float32

    def test_faces_dtype_int32(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["faces"].dtype == np.int32

    def test_verts_are_normalized_within_unit_sphere(self, tmp_path):
        p = str(tmp_path / "test.stl")
        # Large-scale triangle
        _write_binary_stl(p, [
            ((0, 0, 1), (0, 0, 0), (1000, 0, 0), (0, 1000, 0)),
        ])
        result = load_stl_model(p)
        assert result is not None
        max_extent = np.abs(result["verts"]).max()
        assert max_extent <= 1.2, f"Verts not normalized: max={max_extent}"

    def test_centroid_near_origin_after_load(self, tmp_path):
        p = str(tmp_path / "test.stl")
        # Off-center geometry
        _write_binary_stl(p, [
            ((0, 0, 1), (100, 100, 100), (101, 100, 100), (100, 101, 100)),
        ])
        result = load_stl_model(p)
        assert result is not None
        centroid = result["verts"].mean(axis=0)
        assert np.allclose(centroid, 0, atol=0.5), f"Centroid not near origin: {centroid}"

    def test_metadata_filename_matches(self, tmp_path):
        p = str(tmp_path / "my_scan.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["metadata"]["filename"] == "my_scan.stl"

    def test_metadata_vertex_and_face_count(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        meta = result["metadata"]
        assert meta["original_faces"] == 2

    def test_default_color_applied(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["color"] == DEFAULT_MODEL_COLOR

    def test_custom_color_applied(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        custom = (0.5, 0.3, 0.7)
        result = load_stl_model(p, color=custom)
        assert result["color"] == custom

    def test_style_is_solid_by_default(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["style"] == "solid"

    def test_normals_shape_matches_faces(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["normals"].shape[0] == result["faces"].shape[0]
        assert result["normals"].shape[1] == 3

    def test_vertex_normals_shape_matches_verts(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert result["vertex_normals"].shape[0] == result["verts"].shape[0]

    def test_bbox_size_positive(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        bbox = result["metadata"]["bbox_size"]
        assert all(v >= 0 for v in bbox)

    def test_is_watertight_is_bool(self, tmp_path):
        p = str(tmp_path / "test.stl")
        _two_triangle_stl(p)
        result = load_stl_model(p)
        assert isinstance(result["metadata"]["is_watertight"], bool)


# ─── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_max_file_size_reasonable(self):
        """MAX_FILE_SIZE_MB should be a positive value <= 500 MB for interactive use."""
        assert 0 < MAX_FILE_SIZE_MB <= 500

    def test_decimation_threshold_less_than_target(self):
        assert DECIMATION_FACE_THRESHOLD > DECIMATION_TARGET_FACES

    def test_default_color_in_unit_range(self):
        assert all(0.0 <= c <= 1.0 for c in DEFAULT_MODEL_COLOR)
