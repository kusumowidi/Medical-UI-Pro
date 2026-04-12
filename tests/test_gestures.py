"""
tests/test_gestures.py — Unit tests for gesture classification (main.py)

Tests cover finger counting, gesture classification, hand centering,
and edge cases. All tests are headless (no camera, no GUI, no OpenGL).
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Import gesture functions from main without triggering Qt application init
import importlib, sys

# Stub heavy modules before importing main to avoid triggering GUI/camera init
for _mod in ("PyQt6", "PyQt6.QtWidgets", "PyQt6.QtCore", "PyQt6.QtGui",
             "cv2", "mediapipe", "mediapipe.solutions",
             "mediapipe.solutions.hands", "mediapipe.solutions.drawing_utils",
             "gl_renderer"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide a realistic mock for mediapipe.solutions.hands
mp_mock = sys.modules["mediapipe.solutions.hands"]
mp_mock.Hands = MagicMock()
mp_mock.HAND_CONNECTIONS = []

from main import count_fingers, classify_gesture, hand_center, _dist, PINCH_THRESH


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_landmark(x: float, y: float, z: float = 0.0):
    """Return a mock MediaPipe landmark with x/y/z attributes."""
    lm = SimpleNamespace(x=x, y=y, z=z)
    return lm


def _make_hand_landmarks(positions: list):
    """Return a mock hand landmarks object with 21 landmarks.

    Args:
        positions: List of (x, y) tuples. Padded to 21 with zeros if shorter.
    """
    lms = [_make_landmark(*p) for p in positions]
    while len(lms) < 21:
        lms.append(_make_landmark(0.5, 0.5))
    hl = SimpleNamespace(landmark=lms)
    return hl


# ─── Pre-canned hand poses ────────────────────────────────────────────────────

def _open_palm_right():
    """Open palm (Right hand) — all fingers extended upward."""
    # MediaPipe indices: 4=thumb_tip, 3=thumb_ip, 8=index_tip, 6=index_pip, etc.
    positions = [(0.5, 0.5)] * 21
    # Thumb: tip.x < ip.x for Right hand (extended)
    positions[3] = (0.8, 0.5)  # thumb IP — far right
    positions[4] = (0.7, 0.4)  # thumb tip — extended; keep well away from index tip
    # Fingers extended: tip.y < pip.y
    # Index tip at x=0.1 — far from thumb tip at x=0.7 (delta >> PINCH_THRESH * W)
    positions[6] = (0.1, 0.6)   # index pip
    positions[8] = (0.1, 0.2)   # index tip (far left, extended up)
    positions[10] = (0.2, 0.6)  # middle pip
    positions[12] = (0.2, 0.2)  # middle tip
    positions[14] = (0.3, 0.6)  # ring pip
    positions[16] = (0.3, 0.2)  # ring tip
    positions[18] = (0.4, 0.6)  # pinky pip
    positions[20] = (0.4, 0.2)  # pinky tip
    return _make_hand_landmarks(positions)



def _fist_right():
    """Fist (Right hand) — all fingers closed."""
    positions = [(0.5, 0.5)] * 21
    # Thumb folded: tip.x > ip.x for Right hand
    positions[3] = (0.4, 0.5)
    positions[4] = (0.5, 0.5)
    # Fingers closed: tip.y > pip.y
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        positions[pip] = (0.5, 0.3)
        positions[tip] = (0.5, 0.6)
    return _make_hand_landmarks(positions)


def _point_right():
    """Index finger extended, others closed."""
    positions = [(0.5, 0.5)] * 21
    # Thumb folded
    positions[3] = (0.4, 0.5)
    positions[4] = (0.5, 0.5)
    # Index extended
    positions[6] = (0.5, 0.6)
    positions[8] = (0.5, 0.2)
    # Others closed
    for tip, pip in [(12, 10), (16, 14), (20, 18)]:
        positions[pip] = (0.5, 0.3)
        positions[tip] = (0.5, 0.6)
    return _make_hand_landmarks(positions)


def _peace_right():
    """Index + middle extended, others closed."""
    positions = [(0.5, 0.5)] * 21
    # Thumb folded
    positions[3] = (0.4, 0.5)
    positions[4] = (0.5, 0.5)
    # Index + middle extended
    for tip, pip in [(8, 6), (12, 10)]:
        positions[pip] = (0.5, 0.6)
        positions[tip] = (0.5, 0.2)
    # Ring + pinky closed
    for tip, pip in [(16, 14), (20, 18)]:
        positions[pip] = (0.5, 0.3)
        positions[tip] = (0.5, 0.6)
    return _make_hand_landmarks(positions)


def _pinch_right(w=640, h=480):
    """Thumb and index fingertips close together — pinch gesture."""
    positions = [(0.5, 0.5)] * 21
    # Thumb tip and index tip very close (< PINCH_THRESH * w)
    gap = PINCH_THRESH * 0.5  # well within threshold
    positions[4] = (0.5, 0.5)
    positions[8] = (0.5 + gap, 0.5)
    return _make_hand_landmarks(positions)


# ─── _dist ───────────────────────────────────────────────────────────────────

class TestDist:
    def test_zero_distance(self):
        assert _dist((0, 0), (0, 0)) == 0.0

    def test_unit_distance_x(self):
        assert abs(_dist((0, 0), (1, 0)) - 1.0) < 1e-9

    def test_unit_distance_y(self):
        assert abs(_dist((0, 0), (0, 1)) - 1.0) < 1e-9

    def test_pythagorean_345(self):
        assert abs(_dist((0, 0), (3, 4)) - 5.0) < 1e-9

    def test_symmetry(self):
        assert _dist((1, 2), (3, 5)) == _dist((3, 5), (1, 2))


# ─── count_fingers ───────────────────────────────────────────────────────────

class TestCountFingers:
    def test_open_palm_right_returns_5_true(self):
        hl = _open_palm_right()
        f = count_fingers(hl, "Right")
        assert sum(f) >= 4  # At least 4 fingers including thumb

    def test_fist_right_returns_mostly_false(self):
        hl = _fist_right()
        f = count_fingers(hl, "Right")
        assert sum(f) <= 1

    def test_returns_5_element_list(self):
        hl = _make_hand_landmarks([(0.5, 0.5)] * 21)
        f = count_fingers(hl, "Right")
        assert len(f) == 5

    def test_all_elements_are_bool(self):
        hl = _make_hand_landmarks([(0.5, 0.5)] * 21)
        f = count_fingers(hl, "Right")
        assert all(isinstance(x, bool) for x in f)


# ─── classify_gesture ────────────────────────────────────────────────────────

class TestClassifyGesture:
    W, H = 640, 480

    def test_open_palm_classified_as_open(self):
        hl = _open_palm_right()
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert g == "open"

    def test_fist_classified_as_fist(self):
        hl = _fist_right()
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert g == "fist"

    def test_point_gesture_classified_as_point(self):
        hl = _point_right()
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert g == "point"

    def test_peace_gesture_classified_as_peace(self):
        hl = _peace_right()
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert g == "peace"

    def test_pinch_classified_as_pinch(self):
        hl = _pinch_right(self.W, self.H)
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert g == "pinch"

    def test_output_is_string(self):
        hl = _open_palm_right()
        g = classify_gesture(hl, "Right", self.W, self.H)
        assert isinstance(g, str)

    def test_valid_gesture_values(self):
        valid = {"open", "fist", "pinch", "point", "peace", "other"}
        for hl in [_open_palm_right(), _fist_right(), _point_right(), _peace_right()]:
            g = classify_gesture(hl, "Right", self.W, self.H)
            assert g in valid


# ─── hand_center ─────────────────────────────────────────────────────────────

class TestHandCenter:
    W, H = 640, 480

    def test_returns_tuple_of_two_ints(self):
        hl = _make_hand_landmarks([(0.5, 0.5)] * 21)
        cx, cy = hand_center(hl, self.W, self.H)
        assert isinstance(cx, int)
        assert isinstance(cy, int)

    def test_center_within_frame(self):
        hl = _make_hand_landmarks([(0.5, 0.5)] * 21)
        cx, cy = hand_center(hl, self.W, self.H)
        assert 0 <= cx <= self.W
        assert 0 <= cy <= self.H

    def test_all_landmarks_at_top_left(self):
        hl = _make_hand_landmarks([(0.0, 0.0)] * 21)
        cx, cy = hand_center(hl, self.W, self.H)
        assert cx == 0
        assert cy == 0

    def test_all_landmarks_at_bottom_right(self):
        hl = _make_hand_landmarks([(1.0, 1.0)] * 21)
        cx, cy = hand_center(hl, self.W, self.H)
        assert cx == self.W
        assert cy == self.H
