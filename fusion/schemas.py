"""Fusion schemas — shared Pydantic models for aligned multimodal context windows."""

from typing import List, Optional

from pydantic import BaseModel, Field

from audio.schemas import TranscriptSegment
from vision.schemas import Detection


class AlignedFrame(BaseModel):
    """A frame aligned with its corresponding audio transcript segments."""

    frame_idx: int = Field(..., description="Frame index")
    timestamp: Optional[float] = Field(default=None, description="Timestamp in seconds")
    detections: List[Detection] = Field(default_factory=list, description="Vision detections")
    transcripts: List[TranscriptSegment] = Field(default_factory=list, description="Audio segments")
