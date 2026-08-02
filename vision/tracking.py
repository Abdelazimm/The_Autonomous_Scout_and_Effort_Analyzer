"""Multi-object tracking module — assigns persistent IDs to detected entities across frames.

Integrates ByteTrack with a visual ReID gallery: when a frame is supplied to
``update``, player crops are embedded and matched against recently lost tracks
so that transient ByteTrack IDs are re-mapped to stable canonical player IDs
across brief occlusions and tracker resets.
"""

from typing import Optional

import numpy as np
from supervision import ByteTrack, Detections

from vision.reid import DEFAULT_REID_THRESHOLD, FeatureExtractor, ReIDGallery


class PlayerTracker:
    """Wraps ``supervision.ByteTrack`` with optional visual ReID.

    Tuned for soccer footage: a 3-second lost-track buffer reduces ID
    fragmentation when players are briefly occluded, and the matching
    threshold is tightened to avoid ID swaps between nearby players.

    When ``frame`` is passed to :meth:`update`, detected player crops are
    embedded with :class:`FeatureExtractor` and a :class:`ReIDGallery` is
    maintained. New ByteTrack IDs are matched against recently lost tracks via
    cosine similarity; on a match the transient ID is re-mapped to the lost
    track's canonical player ID, keeping IDs stable across occlusions.

    Args:
        lost_track_buffer: Frames to keep a ByteTrack alive after the object is
            no longer detected (3 s @ 30 fps = 90 frames).
        track_activation_threshold: Detection confidence required to start a
            new ByteTrack.
        minimum_matching_threshold: IoU threshold for matching detections to
            existing ByteTracks.
        frame_rate: Source video frame rate.
        reid_threshold: Minimum cosine similarity to accept a ReID match.
        feature_extractor: Optional pre-built :class:`FeatureExtractor`. If
            ``None``, the default MobileNetV3 extractor is built lazily on the
            first frame that triggers ReID.
    """

    def __init__(
        self,
        lost_track_buffer: int = 90,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30,
        reid_threshold: float = DEFAULT_REID_THRESHOLD,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        self._lost_track_buffer = lost_track_buffer
        self._track_activation_threshold = track_activation_threshold
        self._minimum_matching_threshold = minimum_matching_threshold
        self._frame_rate = frame_rate
        self._reid_threshold = reid_threshold
        self._extractor: Optional[FeatureExtractor] = feature_extractor

        self._tracker = ByteTrack(
            lost_track_buffer=lost_track_buffer,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        self._reid = ReIDGallery()
        self._transient_to_canonical: dict = {}
        self._known_transient_ids: set = set()
        self._next_canonical_id = 1
        self._frame_idx = 0

    def update(self, detections: Detections, frame: Optional[np.ndarray] = None) -> Detections:
        """Update the tracker with the given frame's detections.

        When ``frame`` is ``None`` (e.g. the caller only has detections), this
        is a plain ByteTrack pass-through and returns detections with ByteTrack
        ``tracker_id`` values unchanged.

        When ``frame`` is provided, ReID is applied: player crops are embedded,
        new ByteTrack IDs are matched against recently lost tracks, and the
        returned ``tracker_id`` array contains stable canonical player IDs.

        Args:
            detections: Per-frame detections from the detector.
            frame: Optional raw video frame (``HxWx3`` uint8 BGR) used to crop
                players for ReID embeddings.

        Returns:
            Detections augmented with a ``tracker_id`` array (ByteTrack IDs when
            ``frame`` is ``None``, canonical ReID-mapped IDs otherwise).
        """
        if frame is None:
            return self._tracker.update_with_detections(detections)

        prev_transient_ids = set(self._known_transient_ids)
        tracked = self._tracker.update_with_detections(detections)

        # No detections this frame — mark everything lost and advance.
        if tracked.tracker_id is None or len(tracked) == 0:
            for tid in prev_transient_ids:
                canon = self._transient_to_canonical.get(int(tid))
                if canon is not None:
                    self._reid.mark_lost(canon, self._frame_idx)
            self._known_transient_ids = set()
            self._reid.evict(self._frame_idx)
            self._frame_idx += 1
            return tracked

        current_transient_ids = {int(t) for t in tracked.tracker_id}
        lost_transient_ids = prev_transient_ids - current_transient_ids

        # Mark previously-active tracks that disappeared as lost in the gallery.
        for tid in lost_transient_ids:
            canon = self._transient_to_canonical.get(int(tid))
            if canon is not None:
                self._reid.mark_lost(canon, self._frame_idx)

        # Lazily build the default feature extractor on first ReID use.
        if self._extractor is None:
            self._extractor = FeatureExtractor()

        canonical_ids: list = []
        for i, tid in enumerate(tracked.tracker_id):
            tid_int = int(tid)
            if tid_int in self._transient_to_canonical:
                # Continuing track — refresh its gallery embedding.
                canon = self._transient_to_canonical[tid_int]
                crop = self._crop_bbox(frame, tracked.xyxy[i])
                if crop is not None:
                    emb = self._extractor.extract(crop)
                    self._reid.register(canon, emb, self._frame_idx)
                canonical_ids.append(canon)
            else:
                # New ByteTrack ID — try to re-identify against lost tracks.
                crop = self._crop_bbox(frame, tracked.xyxy[i])
                emb = self._extractor.extract(crop) if crop is not None else None
                match = (
                    self._reid.find_best_match(emb, threshold=self._reid_threshold)
                    if emb is not None
                    else None
                )
                if match is not None:
                    canon = match
                else:
                    canon = self._next_canonical_id
                    self._next_canonical_id += 1
                if emb is not None:
                    self._reid.register(canon, emb, self._frame_idx)
                self._transient_to_canonical[tid_int] = canon
                canonical_ids.append(canon)

        self._known_transient_ids = current_transient_ids
        self._reid.evict(self._frame_idx)
        self._frame_idx += 1

        remapped_ids = np.asarray(canonical_ids, dtype=np.int64)
        return self._with_tracker_id(tracked, remapped_ids)

    @staticmethod
    def _with_tracker_id(detections: Detections, tracker_id: np.ndarray) -> Detections:
        """Return a copy of ``detections`` with ``tracker_id`` replaced."""
        return Detections(
            xyxy=detections.xyxy,
            class_id=getattr(detections, "class_id", None),
            confidence=getattr(detections, "confidence", None),
            tracker_id=tracker_id,
            mask=getattr(detections, "mask", None),
            data=getattr(detections, "data", {}),
        )

    @staticmethod
    def _crop_bbox(frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
        """Crop a bounding box from ``frame``, clipping to frame bounds.

        Returns ``None`` for degenerate (zero-area) boxes.
        """
        x1, y1, x2, y2 = (round(float(v)) for v in xyxy)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def reset(self) -> None:
        """Re-instantiate the tracker and ReID gallery to clear all state."""
        self._tracker = ByteTrack(
            lost_track_buffer=self._lost_track_buffer,
            track_activation_threshold=self._track_activation_threshold,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=self._frame_rate,
        )
        self._reid = ReIDGallery()
        self._transient_to_canonical = {}
        self._known_transient_ids = set()
        self._next_canonical_id = 1
        self._frame_idx = 0
        # Keep the feature extractor (avoid re-downloading weights).
