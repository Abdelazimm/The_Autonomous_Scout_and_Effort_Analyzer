"""Multi-object tracking module — assigns persistent IDs to detected entities across frames."""

from supervision import ByteTrack, Detections


class PlayerTracker:
    """Wraps ``supervision.ByteTrack`` to maintain persistent player IDs across frames.

    Uses ``ByteTrack()`` defaults from the supervision >=0.21 API (track_thresh,
    track_buffer, match_thresh, frame_rate are handled internally by the tracker).
    """

    def __init__(self) -> None:
        self._tracker = ByteTrack()

    def update(self, detections: Detections) -> Detections:
        """Assign/update persistent tracker IDs for the given frame's detections.

        Args:
            detections: Per-frame detections from the detector.

        Returns:
            The same detections augmented with a ``tracker_id`` array.
        """
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        """Re-instantiate the underlying tracker to clear all tracking state."""
        self._tracker = ByteTrack()
