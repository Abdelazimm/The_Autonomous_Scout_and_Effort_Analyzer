"""Movement metrics module — computes distance, speed, and effort statistics from tracked trajectories."""

from collections import Counter
from typing import List, Sequence, Tuple

from vision.schemas import MovementLog

# Fixed calibration thresholds (m/s)
IDLE_MAX_SPEED = 1.2
JOGGING_MAX_SPEED = 3.5

# Default pixel-to-meter calibration
DEFAULT_PIXELS_PER_METER = 10.0

# A 2D position in pixels
Position = Tuple[float, float]


def bbox_bottom_center(xyxy: Sequence[float]) -> Position:
    """Return the bottom-center (feet) position of a bbox in supervision ``xyxy`` format.

    Args:
        xyxy: ``(x1, y1, x2, y2)`` bounding box in pixels.

    Returns:
        ``((x1 + x2) / 2, y2)`` — the midpoint of the bottom edge.
    """
    x1, _y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, y2)


def euclidean_distance(p1: Position, p2: Position) -> float:
    """Euclidean distance between two 2D points (pixels)."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def speed_mps(
    prev_pos: Position,
    curr_pos: Position,
    dt_s: float,
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
) -> float:
    """Speed in meters per second between two positions separated by ``dt_s`` seconds.

    Args:
        prev_pos: Previous position in pixels.
        curr_pos: Current position in pixels.
        dt_s: Time delta between the two positions in seconds.
        pixels_per_meter: Pixel-to-meter calibration constant.

    Returns:
        Speed in m/s. Returns ``0.0`` when ``dt_s <= 0`` to avoid division by zero.
    """
    if dt_s <= 0:
        return 0.0
    dist_px = euclidean_distance(prev_pos, curr_pos)
    return (dist_px / pixels_per_meter) / dt_s


def classify_state(speed_mps_value: float) -> str:
    """Classify a speed value into ``idle`` / ``jogging`` / ``sprinting``.

    Thresholds (fixed calibration):
        - ``speed < 1.2 m/s``           -> ``"idle"``
        - ``1.2 <= speed < 3.5 m/s``    -> ``"jogging"``
        - ``speed >= 3.5 m/s``          -> ``"sprinting"``
    """
    if speed_mps_value < IDLE_MAX_SPEED:
        return "idle"
    if speed_mps_value < JOGGING_MAX_SPEED:
        return "jogging"
    return "sprinting"


def generate_movement_log(
    player_id: int,
    positions: Sequence[Position],
    timestamps: Sequence[float],
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER,
    juggle_count: int = 0,
) -> MovementLog:
    """Aggregate a player's per-frame trajectory into a window-level ``MovementLog``.

    The window's ``state`` is the most frequent per-frame classified state
    (dominant state), ``avg_speed_mps`` is the mean of per-frame speeds, and
    ``distance_covered_m`` is the sum of per-frame displacements in meters.

    Args:
        player_id: Persistent tracker ID for the player.
        positions: Ordered sequence of ``(x, y)`` positions in pixels (>=1 point).
        timestamps: Ordered sequence of timestamps in seconds (same length as ``positions``).
        pixels_per_meter: Pixel-to-meter calibration constant.
        juggle_count: Optional ball-juggle count for the window.

    Returns:
        A ``MovementLog`` describing the player's movement over the window.

    Raises:
        ValueError: If ``positions`` and ``timestamps`` have different lengths.
    """
    if len(positions) != len(timestamps):
        raise ValueError("positions and timestamps must have the same length")

    window_start = timestamps[0] if timestamps else 0.0
    window_end = timestamps[-1] if timestamps else 0.0

    if len(positions) < 2:
        # Not enough data to compute displacement-based speed
        return MovementLog(
            player_id=player_id,
            window_start_s=window_start,
            window_end_s=window_end,
            state="idle",
            avg_speed_mps=0.0,
            distance_covered_m=0.0,
            juggle_count=juggle_count,
        )

    speeds: List[float] = []
    total_dist_m = 0.0
    for i in range(1, len(positions)):
        dt = timestamps[i] - timestamps[i - 1]
        dist_px = euclidean_distance(positions[i - 1], positions[i])
        total_dist_m += dist_px / pixels_per_meter
        speeds.append(speed_mps(positions[i - 1], positions[i], dt, pixels_per_meter))

    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    states = [classify_state(s) for s in speeds]
    dominant_state = Counter(states).most_common(1)[0][0]

    return MovementLog(
        player_id=player_id,
        window_start_s=window_start,
        window_end_s=window_end,
        state=dominant_state,
        avg_speed_mps=avg_speed,
        distance_covered_m=total_dist_m,
        juggle_count=juggle_count,
    )
