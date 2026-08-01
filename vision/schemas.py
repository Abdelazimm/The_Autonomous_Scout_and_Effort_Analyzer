"""Shared Pydantic schemas for detections and ground-truth frames across the pipeline."""

from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box in YOLO format (center_x, center_y, width, height)."""

    x: float = Field(..., description="Normalized center X coordinate [0, 1]")
    y: float = Field(..., description="Normalized center Y coordinate [0, 1]")
    w: float = Field(..., description="Normalized width [0, 1]")
    h: float = Field(..., description="Normalized height [0, 1]")


class Detection(BaseModel):
    """A single object detection within a frame."""

    frame_idx: int = Field(..., description="Index of the frame this detection belongs to")
    class_id: int = Field(..., description="Class ID from the unified taxonomy")
    class_name: str = Field(..., description="Human-readable class name")
    bbox: BoundingBox = Field(..., description="Normalized bounding box")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")


class GroundTruthFrame(BaseModel):
    """A ground-truth or predicted frame containing multiple detections."""

    frame_idx: int = Field(..., description="Index of the frame")
    timestamp: Optional[float] = Field(default=None, description="Timestamp in seconds (if available)")
    detections: List[Detection] = Field(default_factory=list, description="Detections in this frame")
