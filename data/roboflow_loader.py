"""Roboflow data loader — parses YOLO-format label files into Pydantic GroundTruthFrame objects."""

from pathlib import Path
from typing import Dict, Optional

from vision.schemas import BoundingBox, Detection, GroundTruthFrame

# Default class map — adopted from the 4 Roboflow classes
DEFAULT_CLASS_MAP: Dict[int, str] = {
    0: "ball",
    1: "player",
    2: "referee",
    3: "goalkeeper",
}


def load_roboflow_yolo(
    label_path: str,
    class_map: Optional[Dict[int, str]] = None,
    frame_idx: int = 0,
    timestamp: Optional[float] = None,
) -> GroundTruthFrame:
    """Parse a Roboflow YOLO-format ``.txt`` label file into a ``GroundTruthFrame``.

    Each line in the label file follows the YOLO format::

        <class_id> <x_center> <y_center> <width> <height>

    All coordinates are normalized to [0, 1].

    Args:
        label_path: Path to the YOLO ``.txt`` label file.
        class_map: Mapping of class IDs to class names. Defaults to the
            unified taxonomy (ball, player, referee, goalkeeper).
        frame_idx: Frame index to assign. Defaults to 0.
        timestamp: Optional timestamp in seconds.

    Returns:
        A ``GroundTruthFrame`` containing all parsed detections.
    """
    if class_map is None:
        class_map = DEFAULT_CLASS_MAP

    label_file = Path(label_path)
    detections: list[Detection] = []

    if label_file.exists():
        for line in label_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            x, y, w, h = (float(v) for v in parts[1:5])
            cls_name = class_map.get(cls_id, f"unknown_{cls_id}")

            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=1.0,
                )
            )

    return GroundTruthFrame(
        frame_idx=frame_idx,
        timestamp=timestamp,
        detections=detections,
    )
