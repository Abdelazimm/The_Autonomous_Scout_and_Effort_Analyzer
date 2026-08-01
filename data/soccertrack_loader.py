"""SoccerTrack data loader — wraps SoccerTrack's ``load_match()`` output into Pydantic GroundTruthFrame objects."""

from typing import Any, Dict, Optional

from vision.schemas import BoundingBox, Detection, GroundTruthFrame

# Default class map — adopted from the 4 Roboflow classes
DEFAULT_CLASS_MAP: Dict[int, str] = {
    0: "ball",
    1: "player",
    2: "referee",
    3: "goalkeeper",
}


def load_soccertrack_frame(
    match_data: Any,
    frame_idx: int = 0,
    class_map: Optional[Dict[int, str]] = None,
    fps: float = 30.0,
) -> GroundTruthFrame:
    """Convert a single frame from SoccerTrack's ``load_match()`` output into a ``GroundTruthFrame``.

    SoccerTrack's ``load_match()`` helper returns a DataFrame where each row
    represents a tracked object with columns typically including::

        ``frame``, ``track_id``, ``x``, ``y``, ``w``, ``h``, ``class_id``

    This function filters the DataFrame for the given ``frame_idx``, normalizes
    bounding boxes, and maps class IDs to the unified taxonomy.

    Args:
        match_data: A pandas DataFrame (or dict-like) from SoccerTrack's
            ``load_match()`` containing per-frame tracking data.
        frame_idx: The frame index to extract. Defaults to 0.
        class_map: Mapping of class IDs to class names. Defaults to the
            unified taxonomy (ball, player, referee, goalkeeper).
        fps: Frames per second — used to compute a timestamp. Defaults to 30.

    Returns:
        A ``GroundTruthFrame`` containing all detections for the requested frame.
    """
    if class_map is None:
        class_map = DEFAULT_CLASS_MAP

    detections: list[Detection] = []

    # Filter for the requested frame
    try:
        frame_rows = match_data[match_data["frame"] == frame_idx]
    except Exception:
        # If match_data is a plain dict of lists, build a simple filter
        if isinstance(match_data, dict):
            import pandas as pd

            match_data = pd.DataFrame(match_data)
            frame_rows = match_data[match_data["frame"] == frame_idx]
        else:
            raise

    for _, row in frame_rows.iterrows():
        cls_id = int(row.get("class_id", 1))
        cls_name = class_map.get(cls_id, f"unknown_{cls_id}")

        # SoccerTrack coordinates may be in pixels — normalize to [0, 1]
        # using image dimensions if available, otherwise assume already normalized
        x = float(row.get("x", 0.5))
        y = float(row.get("y", 0.5))
        w = float(row.get("w", 0.1))
        h = float(row.get("h", 0.1))

        img_w = row.get("img_width")
        img_h = row.get("img_height")
        if img_w and img_h:
            x /= float(img_w)
            w /= float(img_w)
            y /= float(img_h)
            h /= float(img_h)

        detections.append(
            Detection(
                frame_idx=frame_idx,
                class_id=cls_id,
                class_name=cls_name,
                bbox=BoundingBox(x=x, y=y, w=w, h=h),
                confidence=float(row.get("confidence", 1.0)),
            )
        )

    return GroundTruthFrame(
        frame_idx=frame_idx,
        timestamp=frame_idx / fps,
        detections=detections,
    )
