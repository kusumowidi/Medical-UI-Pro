# Changelog

All notable changes to **Medical UI Pro — Holographic Edition** are documented here.  
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2026-04-13

### Added
- Hardware-accelerated OpenGL 3.3 renderer (`gl_renderer.py`) with GLSL 330 shaders
  - Phong shading with 3-point lighting (key + fill + back)
  - Rim lighting (Fresnel-style edge glow)
  - X-ray volumetric mode (additive alpha blending)
  - Wireframe mode (depth-graded)
  - Orientation gizmo (XYZ axes) with QPainter label overlay
  - Spinning loading indicator
  - Persistent FPS counter at viewport bottom-left
- Collapsible sidebar (Tab key / collapse button)
- Scrollable sidebar with overflow support for small screens
- Dark/Light theme toggle (radiology dark vs OR light)
- Dark theme: `style.qss` / Light theme: `style_light.qss`
- Anatomical view presets: ANT, POST, L, R, SUP, INF
- Material color picker with 7 clinical presets
  (Bone, Tissue, Organ, Vessel, Cartilage, Metal Implant, Nerve)
- Scan metadata panel: vertex/face count, bounding box, surface area, volume, watertight
- Screenshot export (Ctrl+S / "CAPTURE VIEWPORT" button) — PNG output
- Two-way slider binding: sliders reflect live view state changes from gestures
- Auto mesh decimation: meshes > 50,000 faces simplified via quadric decimation
- Per-vertex normal computation for smooth OpenGL shading
- Loading progress bar + animated spinner (software renderer)

### Changed
- Refactored from single-file procedural script to modular architecture:
  - `main.py` — application window and gesture engine
  - `gl_renderer.py` — OpenGL viewport
  - `renderer.py` — CPU software renderer fallback
  - `models.py` — STL loader, validation, metadata
- `HandTracker` moved to background daemon thread with lock-free queue
- Gesture state machine now handles dual-hand pan (two open palms)
- Status bar redesigned with FPS, mesh info, render mode, camera state, engine indicator

### Fixed
- Software renderer correctly handles meshes with > 15,000 faces (point-cloud fallback)
- OpenGL renderer state correctly restored after X-ray (blend / depth mask) and wireframe (polygon mode)
- `AppViewer` closes camera capture on window close event (no zombie thread)

---

## [1.0.0] — 2026-04-07

### Added
- Initial procedural 3D STL viewer with software renderer
- Mouse controls (left-drag rotate, right-drag pan, scroll zoom)
- MediaPipe gesture control (open palm rotate, pinch zoom, point pan, peace reset)
- Basic PyQt6 window with camera PIP feed
- STL loading via trimesh with basic mesh centering
- Dark mode QSS stylesheet

---

[2.0.0]: https://github.com/your-org/3d-control/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/your-org/3d-control/releases/tag/v1.0.0
