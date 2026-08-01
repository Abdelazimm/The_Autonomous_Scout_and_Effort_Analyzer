"""Unit tests for the Roboflow YOLO-format data loader."""

import pytest

from data.roboflow_loader import load_roboflow_yolo
from vision.schemas import GroundTruthFrame


@pytest.fixture
def mock_yolo_label(tmp_path):
    """Create a temporary YOLO-format label file with 3 detections."""
    label_content = (
        "0 0.50 0.40 0.02 0.02\n"   # ball
        "1 0.30 0.60 0.05 0.10\n"   # player
        "3 0.70 0.20 0.05 0.10\n"   # goalkeeper
    )
    label_file = tmp_path / "frame_0001.txt"
    label_file.write_text(label_content)
    return str(label_file)


def test_load_roboflow_yolo_parses_correct_number_of_detections(mock_yolo_label):
    """The loader should return exactly 3 detections from a 3-line label file."""
    frame = load_roboflow_yolo(mock_yolo_label, frame_idx=1)
    assert isinstance(frame, GroundTruthFrame)
    assert len(frame.detections) == 3


def test_load_roboflow_yolo_correct_class_names(mock_yolo_label):
    """Class names should match the unified taxonomy via the class_map."""
    frame = load_roboflow_yolo(mock_yolo_label, frame_idx=1)
    class_names = [d.class_name for d in frame.detections]
    assert class_names == ["ball", "player", "goalkeeper"]


def test_load_roboflow_yolo_correct_bbox_values(mock_yolo_label):
    """Bounding box values should be parsed exactly as written in the label file."""
    frame = load_roboflow_yolo(mock_yolo_label, frame_idx=1)
    ball = frame.detections[0]
    assert ball.class_id == 0
    assert ball.bbox.x == pytest.approx(0.50)
    assert ball.bbox.y == pytest.approx(0.40)
    assert ball.bbox.w == pytest.approx(0.02)
    assert ball.bbox.h == pytest.approx(0.02)


def test_load_roboflow_yolo_frame_idx_and_timestamp(mock_yolo_label):
    """Frame index and timestamp should be propagated correctly."""
    frame = load_roboflow_yolo(mock_yolo_label, frame_idx=42, timestamp=1.4)
    assert frame.frame_idx == 42
    assert frame.timestamp == 1.4


def test_load_roboflow_yolo_empty_file(tmp_path):
    """An empty label file should produce a frame with zero detections."""
    label_file = tmp_path / "empty.txt"
    label_file.write_text("")
    frame = load_roboflow_yolo(str(label_file))
    assert len(frame.detections) == 0


def test_load_roboflow_yolo_nonexistent_file():
    """A nonexistent label file should produce an empty frame (no crash)."""
    frame = load_roboflow_yolo("/nonexistent/path/labels.txt")
    assert len(frame.detections) == 0


def test_load_roboflow_yolo_custom_class_map(tmp_path):
    """A custom class_map should override the default taxonomy."""
    label_file = tmp_path / "custom.txt"
    label_file.write_text("5 0.1 0.2 0.3 0.4\n")
    custom_map = {5: "coach"}
    frame = load_roboflow_yolo(str(label_file), class_map=custom_map)
    assert frame.detections[0].class_name == "coach"
    assert frame.detections[0].class_id == 5
