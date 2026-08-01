"""Audio schemas — shared Pydantic models for transcription segments and aligned timestamps."""

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single transcribed segment from the audio track."""

    text: str = Field(..., description="Transcribed text")
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    confidence: Optional[float] = Field(default=None, description="Transcription confidence score")
