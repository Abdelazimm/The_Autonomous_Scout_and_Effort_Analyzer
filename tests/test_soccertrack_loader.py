"""Unit tests for the SoccerTrack data loader."""

import pytest

from data.soccertrack_loader import load_soccertrack_frame
from vision.schemas import GroundTruthFrame


@pytest.fixture
def mock_match_data():
    """Create a mock DataFrame mimicking SoccerTrack's load_match() output."""
    import pandas as pd

    data = {
        "frame": [0, 0, 0, 1, 1],
        "track_id": [1, 2, 3, 1, 2],
        "x": [0.50, 0.30, 0.70, 0.52, 0.32],
        "y": [0.40, 0.60, 0.20, 0.42, 0.62],
        "w": [0.02, 0.05, 0.05, 0.02, 0.05],
        "h": [0.02, 0.10, 0.10, 0.02, 0.10],
        "class_id": [0, 1, 3, 0, 1],
        "confidence": [0.95, 0.88, 0.91, 0.93, 0.85],
    }
    return pd.DataFrame(data)


def test_load_soccertrack_frame_returns_ground_truth_frame(mock_match_data):
    """The loader should return a valid GroundTruthFrame."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0)
    assert isinstance(frame, GroundTruthFrame)


def test_load_soccertrack_frame_correct_detections(mock_match_data):
    """Frame 0 should contain exactly 3 detections (ball, player, goalkeeper)."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0)
    assert len(frame.detections) == 3
    class_names = [d.class_name for d in frame.detections]
    assert class_names == ["ball", "player", "goalkeeper"]


def test_load_soccertrack_frame_correct_bbox(mock_match_data):
    """Bounding box values should match the mock data."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0)
    ball = frame.detections[0]
    assert ball.class_id == 0
    assert ball.bbox.x == pytest.approx(0.50)
    assert ball.bbox.y == pytest.approx(0.40)
    assert ball.bbox.w == pytest.approx(0.02)
    assert ball.bbox.h == pytest.approx(0.02)


def test_load_soccertrack_frame_confidence(mock_match_data):
    """Confidence values should be propagated from the mock data."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0)
    assert frame.detections[0].confidence == pytest.approx(0.95)
    assert frame.detections[1].confidence == pytest.approx(0.88)


def test_load_soccertrack_frame_timestamp(mock_match_data):
    """Timestamp should be computed as frame_idx / fps."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0, fps=30.0)
    assert frame.timestamp == pytest.approx(0.0)

    frame_1 = load_soccertrack_frame(mock_match_data, frame_idx=1, fps=30.0)
    assert frame_1.timestamp == pytest.approx(1 / 30.0)


def test_load_soccertrack_frame_different_frame(mock_match_data):
    """Frame 1 should contain exactly 2 detections."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=1)
    assert len(frame.detections) == 2
    assert frame.detections[0].class_name == "ball"
    assert frame.detections[1].class_name == "player"


def test_load_soccertrack_frame_empty_frame(mock_match_data):
    """A frame index with no data should produce zero detections."""
    frame = load_soccertrack_frame(mock_match_data, frame_idx=999)
    assert len(frame.detections) == 0


def test_load_soccertrack_frame_custom_class_map(mock_match_data):
    """A custom class_map should override the default taxonomy."""
    custom_map = {0: "soccer_ball", 1: "athlete", 3: "goalie"}
    frame = load_soccertrack_frame(mock_match_data, frame_idx=0, class_map=custom_map)
    assert frame.detections[0].class_name == "soccer_ball"
    assert frame.detections[2].class_name == "goalie"
