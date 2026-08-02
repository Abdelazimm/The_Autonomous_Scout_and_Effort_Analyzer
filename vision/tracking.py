"""Multi-object tracking module — assigns persistent IDs to detected entities across frames."""

from supervision import ByteTrack, Detections


class PlayerTracker:
    """Wraps ``supervision.ByteTrack`` to maintain persistent player IDs across frames.

    Args:
        track_thresh: Detection confidence threshold for track activation.
        track_buffer: Number of frames to buffer when a track is lost.
        match_thresh: Threshold for matching tracks with detections.
        frame_rate: Frame rate of the source video.
    """

    def __init__(
        self,
        track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._frame_rate = frame_rate
        self._tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

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
        self._tracker = ByteTrack(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            frame_rate=self._frame_rate,
        )
