"""Multi-object tracking module — assigns persistent IDs to detected entities across frames."""

from supervision import ByteTrack, Detections


class PlayerTracker:
    """Wraps ``supervision.ByteTrack`` to maintain persistent player IDs across frames.

    Tuned for soccer footage: a 3-second lost-track buffer reduces ID
    fragmentation when players are briefly occluded, and the matching
    threshold is tightened to avoid ID swaps between nearby players.

    Args:
        lost_track_buffer: Frames to keep a track alive after the object is
            no longer detected (3 s @ 30 fps = 90 frames).
        track_activation_threshold: Detection confidence required to start
            a new track.
        minimum_matching_threshold: IoU threshold for matching detections
            to existing tracks.
        frame_rate: Source video frame rate (used to convert the buffer to
            frames).
    """

    def __init__(
        self,
        lost_track_buffer: int = 90,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30,
    ) -> None:
        self._lost_track_buffer = lost_track_buffer
        self._track_activation_threshold = track_activation_threshold
        self._minimum_matching_threshold = minimum_matching_threshold
        self._frame_rate = frame_rate
        self._tracker = ByteTrack(
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
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
            lost_track_buffer=self._lost_track_buffer,
            track_activation_threshold=self._track_activation_threshold,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=self._frame_rate,
        )
