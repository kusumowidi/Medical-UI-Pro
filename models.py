"""
models.py — Medical 3D Model Loader

Handles STL file loading with mesh validation, decimation, metadata extraction,
and per-vertex normal computation for smooth OpenGL shading.

Mesh Dict Contract
------------------
All functions that return mesh data produce a dictionary with the following schema:

    {
        "verts":          np.ndarray,   # (N, 3) float32 — centered, normalized to [-1, 1]
        "faces":          np.ndarray,   # (M, 3) int32   — triangle vertex indices
        "normals":        np.ndarray,   # (M, 3) float32 — face unit normals
        "vertex_normals": np.ndarray,   # (N, 3) float32 — smooth per-vertex normals
        "color":          tuple,        # (r, g, b) floats in [0.0, 1.0]
        "style":          str,          # "solid" | "wireframe" | "mixed" | "pointcloud"
        "metadata":       dict,         # see load_stl_model docstring for full schema
    }

The ``style`` field controls how the software renderer (renderer.py) draws the mesh:

- ``"solid"``      — filled triangles with Phong shading (default)
- ``"wireframe"``  — triangle edges only
- ``"mixed"``      — filled + wireframe overlay
- ``"pointcloud"`` — vertex dots only (auto-selected for very large meshes)

Note: GLViewport (gl_renderer.py) does not use the ``style`` field; render mode
is instead controlled via ``set_wireframe()`` and the X-ray toggle.
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB: float = 200.0
"""Soft size limit (MB) before prompting the user for confirmation.

Files above this threshold are still loadable but may cause significant RAM pressure.
Typical clinical STL exports from CT segmentation software range from 5–80 MB.
Set higher only if your target system has ≥ 16 GB of available RAM.
"""

DECIMATION_FACE_THRESHOLD: int = 50_000
"""Face count above which quadric decimation is applied before rendering.

Meshes with more than this many faces are simplified to DECIMATION_TARGET_FACES
using trimesh's simplify_quadric_decimation for interactive performance.
"""

DECIMATION_TARGET_FACES: int = 30_000
"""Target face count after quadric decimation (see DECIMATION_FACE_THRESHOLD)."""

DEFAULT_MODEL_COLOR: Tuple[float, float, float] = (0.85, 0.85, 0.85)
"""Default mesh color (light grey) applied when no color is specified by the caller."""


def get_file_size_mb(filepath: str) -> float:
    """Return the file size in megabytes."""
    try:
        return os.path.getsize(filepath) / (1024 * 1024)
    except OSError:
        return 0.0


def _compute_vertex_normals(
    verts: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
) -> np.ndarray:
    """Compute smooth per-vertex normals by averaging adjacent face normals.

    For each vertex, accumulates the unit normals of all faces that share it,
    then re-normalizes. This produces smooth Gouraud/Phong shading across
    curved surfaces without requiring duplicated vertices.

    The 1e-9 epsilon prevents division-by-zero for isolated vertices that
    appear in no face (e.g., after decimation artifacts).

    Args:
        verts: (N, 3) array of vertex positions.
        faces: (M, 3) array of triangle vertex indices.
        face_normals: (M, 3) array of pre-computed, unit face normals.

    Returns:
        (N, 3) float32 array of per-vertex unit normals. Values are in [-1, 1].
        Isolated vertices (not referenced by any face) will have a zero normal.
    """
    vertex_normals = np.zeros_like(verts, dtype=np.float64)
    # Accumulate face normals at each vertex using unbuffered indexing
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    # Re-normalize accumulated sums to unit length
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / (norms + 1e-9)
    return vertex_normals.astype(np.float32)


def load_stl_model(
    filepath: str,
    color: Tuple[float, float, float] = DEFAULT_MODEL_COLOR,
    max_faces: int = DECIMATION_FACE_THRESHOLD,
    target_faces: int = DECIMATION_TARGET_FACES,
) -> Optional[dict]:
    """Load an external STL file and prepare it for the rendering pipeline.

    Processing pipeline:
        1. Validate file existence and log file size.
        2. Load with trimesh; concatenate if Scene (multi-body STL).
        3. Extract metadata (filename, counts, bounding box, area, volume).
        4. Apply quadric decimation if face count exceeds ``max_faces``.
        5. Center geometry at the origin and normalize to fit in [-1, 1].
        6. Compute face normals and smooth per-vertex normals.

    Args:
        filepath:     Absolute or relative path to the ``.stl`` file.
        color:        Initial mesh color as (r, g, b) floats in [0.0, 1.0].
                      Defaults to DEFAULT_MODEL_COLOR (light grey).
        max_faces:    Face count threshold above which decimation is applied.
        target_faces: Target face count after decimation.

    Returns:
        A mesh dict conforming to the Mesh Dict Contract (see module docstring),
        or ``None`` if the file cannot be loaded (not found, corrupt, out-of-memory).

    Metadata dict keys:
        filename (str):          Base filename of the loaded file.
        original_verts (int):    Vertex count before decimation.
        original_faces (int):    Face count before decimation.
        bbox_size (list[float]): [width, height, depth] in original mesh units.
        surface_area (float):    Surface area in original mesh units.
        volume (float | None):   Volume if watertight; None otherwise.
        is_watertight (bool):    Whether the mesh is a closed, manifold solid.
    """
    logger.info("Loading STL: %s", filepath)

    if not os.path.isfile(filepath):
        logger.error("File not found: %s", filepath)
        return None

    file_size_mb = get_file_size_mb(filepath)
    logger.info("File size: %.1f MB", file_size_mb)

    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning("Large file (%.1f MB). May consume significant RAM.", file_size_mb)

    try:
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if not hasattr(mesh, 'vertices') or len(mesh.vertices) == 0:
            logger.error("STL contains no geometry: %s", filepath)
            return None

        # ── Collect metadata before any modifications ─────────────────────
        raw_verts = mesh.vertices
        raw_faces = mesh.faces
        bbox_min = raw_verts.min(axis=0)
        bbox_max = raw_verts.max(axis=0)
        bbox_size = bbox_max - bbox_min

        metadata = {
            "filename": os.path.basename(filepath),
            "original_verts": int(len(raw_verts)),
            "original_faces": int(len(raw_faces)),
            "bbox_size": bbox_size.tolist(),
            "surface_area": float(mesh.area) if hasattr(mesh, 'area') else 0.0,
            "volume": float(mesh.volume) if (hasattr(mesh, 'is_watertight') and mesh.is_watertight) else None,
            "is_watertight": bool(mesh.is_watertight) if hasattr(mesh, 'is_watertight') else False,
        }

        logger.info("Loaded: %d verts, %d faces", len(raw_verts), len(raw_faces))

        verts = mesh.vertices.copy()
        faces = mesh.faces.copy()

        # ── Decimation ────────────────────────────────────────────────────
        if len(faces) > max_faces:
            logger.info("Decimating %d → ~%d faces...", len(faces), target_faces)
            try:
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if simplified is not None and len(simplified.faces) > 0:
                    verts = simplified.vertices.copy()
                    faces = simplified.faces.copy()
                    logger.info("Decimated: %d verts, %d faces", len(verts), len(faces))
                else:
                    logger.warning("Decimation failed; using original.")
            except Exception as e:
                logger.warning("Decimation error: %s. Using original.", e)

        # ── Center and normalize ──────────────────────────────────────────
        centroid = verts.mean(axis=0)
        verts -= centroid
        scale = np.abs(verts).max()
        if scale > 0:
            verts /= (scale * 0.95)

        # ── Face normals ──────────────────────────────────────────────────
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        fn_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / (fn_norms + 1e-9)

        # ── Vertex normals (for smooth GL shading) ────────────────────────
        vertex_normals = _compute_vertex_normals(verts, faces, face_normals)

        result = {
            "verts": verts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "normals": face_normals.astype(np.float32),
            "vertex_normals": vertex_normals,
            "color": color,
            "style": "solid",
            "metadata": metadata,
        }
        logger.info("STL ready: %s", metadata["filename"])
        return result

    except MemoryError:
        logger.error("Out of memory loading %s (%.1f MB).", filepath, file_size_mb)
        return None
    except Exception as e:
        logger.error("Failed to load STL '%s': %s", filepath, e, exc_info=True)
        return None