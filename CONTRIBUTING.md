# Contributing to Medical UI Pro

Thank you for your interest in contributing! This document covers how to set up the development environment, our coding conventions, and the pull request process.

> ⚠️ **Note:** This project is for educational and research use only. Contributions that imply clinical diagnostic capability will not be accepted without appropriate regulatory context.

---

## 📋 Table of Contents

1. [Development Setup](#development-setup)
2. [Branch Strategy](#branch-strategy)
3. [Coding Conventions](#coding-conventions)
4. [Running Tests](#running-tests)
5. [Submitting a Pull Request](#submitting-a-pull-request)
6. [Reporting Bugs](#reporting-bugs)

---

## Development Setup

### Requirements

- Python 3.9–3.12
- A webcam (optional — mouse fallback available)
- GPU with OpenGL 3.3+ (optional — software renderer fallback)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/3d-control.git
cd 3d-control

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install runtime + dev dependencies
pip install -r requirements-lock.txt
pip install ruff mypy

# 4. Run the application
python main.py
```

---

## Branch Strategy

We use a simple trunk-based workflow:

| Branch | Purpose |
|---|---|
| `main` | Stable, release-ready code |
| `dev` | Active development integration |
| `feature/<name>` | New feature branches — branch off `dev` |
| `fix/<name>` | Bug fix branches — branch off `main` for hotfixes |

**Do not commit directly to `main`.**

---

## Coding Conventions

### Python Style

- **Formatter / linter:** [`ruff`](https://docs.astral.sh/ruff/) — run `ruff check .` before committing
- **Type checking:** [`mypy`](https://mypy.readthedocs.io/) — run `mypy main.py models.py`
- **Line length:** 100 characters max
- **String quotes:** Double quotes preferred

### Docstrings

All public classes and functions must have a docstring following Google style:

```python
def load_stl_model(filepath: str, color: tuple = ...) -> Optional[dict]:
    """Load an STL file and prepare it for the rendering pipeline.

    Args:
        filepath: Absolute path to the .stl file.
        color: RGB tuple (0.0–1.0) for initial mesh color.

    Returns:
        A mesh dict with keys: verts, faces, normals, vertex_normals,
        color, style, metadata. Returns None on failure.
    """
```

### Comments

- Avoid `(#N)` magic number references — use descriptive section comments instead
- GLSL shader code must include a comment for every non-trivial math block
- Constants must include a docstring explaining the rationale for the value

### Mesh Dict Contract

All mesh data passed between modules must conform to this schema:

```python
{
    "verts":          np.ndarray,   # shape (N, 3), float32
    "faces":          np.ndarray,   # shape (M, 3), int32
    "normals":        np.ndarray,   # shape (M, 3), float32 — face normals
    "vertex_normals": np.ndarray,   # shape (N, 3), float32 — smooth shading
    "color":          tuple,        # (r, g, b) floats 0.0–1.0
    "style":          str,          # "solid" | "wireframe" | "mixed" | "pointcloud"
    "metadata":       dict,         # see models.py for full schema
}
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run linter
ruff check .

# Run type checker
mypy main.py models.py renderer.py gl_renderer.py
```

---

## Submitting a Pull Request

1. **Fork** the repository and create your branch from `dev`
2. **Write tests** for any new functionality in `tests/`
3. **Run the test suite** — all tests must pass
4. **Run `ruff check .`** — zero linting errors
5. **Update `CHANGELOG.md`** — add your change under `[Unreleased]`
6. **Open a PR** with a clear title and description linking to any relevant issues
7. **Request review** from at least one maintainer

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Motivation
Why is this change needed?

## Changes
- List of specific changes made

## Testing
How was this tested?

## Checklist
- [ ] Tests added / updated
- [ ] CHANGELOG.md updated
- [ ] ruff passes
- [ ] mypy passes
- [ ] Documentation updated
```

---

## Reporting Bugs

Please open a GitHub Issue with the following information:

- **OS and Python version**
- **GPU / OpenGL version** (run `python -c "from OpenGL.GL import *; print(glGetString(GL_VERSION))"`)
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Relevant log output** (the terminal prints timestamped logs)

---

*Questions? Open a discussion on the GitHub repository.*
