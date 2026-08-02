"""Unit tests for movement metrics — speed, state classification, and window aggregation."""

import pytest
from pydantic import ValidationError

from vision.movement_metrics import (
    DEFAULT_PIXELS_PER_METER,
    MAX_HUMAN_SPEED_MPS,
    bbox_bottom_center,
    classify_state,
    euclidean_distance,
    generate_movement_log,
    smooth_positions,
    speed_mps,
)
from vision.schemas import MovementLog

# ── bbox_bottom_center ─────────────────────────────────────────────


def test_bbox_bottom_center_returns_feet_position():
    """Bottom-center should be the midpoint of the bottom edge of the bbox."""
    pos = bbox_bottom_center((10.0, 20.0, 30.0, 80.0))
    assert pos == (20.0, 80.0)


def test_bbox_bottom_center_square():
    """A square bbox should return the center-x of its bottom edge."""
    pos = bbox_bottom_center((0.0, 0.0, 100.0, 100.0))
    assert pos == (50.0, 100.0)


# ── euclidean_distance ─────────────────────────────────────────────


def test_euclidean_distance_horizontal():
    """Purely horizontal displacement equals the x delta."""
    assert euclidean_distance((0.0, 0.0), (10.0, 0.0)) == pytest.approx(10.0)


def test_euclidean_distance_vertical():
    """Purely vertical displacement equals the y delta."""
    assert euclidean_distance((5.0, 0.0), (5.0, 24.0)) == pytest.approx(24.0)


def test_euclidean_distance_pythagorean():
    """A 3-4-5 triangle should yield distance 5."""
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_euclidean_distance_zero():
    """Identical points should yield zero distance."""
    assert euclidean_distance((7.0, 7.0), (7.0, 7.0)) == pytest.approx(0.0)


# ── speed_mps ──────────────────────────────────────────────────────


def test_speed_mps_known_value():
    """100 px over 1 s at 10 px/m should yield 10 m/s."""
    speed = speed_mps((0.0, 0.0), (100.0, 0.0), dt_s=1.0, pixels_per_meter=10.0)
    assert speed == pytest.approx(10.0)


def test_speed_mps_custom_calibration():
    """200 px over 2 s at 20 px/m should yield 5 m/s."""
    speed = speed_mps((0.0, 0.0), (200.0, 0.0), dt_s=2.0, pixels_per_meter=20.0)
    assert speed == pytest.approx(5.0)


def test_speed_mps_zero_displacement():
    """No displacement should yield zero speed."""
    speed = speed_mps((50.0, 50.0), (50.0, 50.0), dt_s=1.0)
    assert speed == pytest.approx(0.0)


def test_speed_mps_zero_dt_returns_zero():
    """A non-positive time delta should be guarded to avoid division by zero."""
    assert speed_mps((0.0, 0.0), (100.0, 0.0), dt_s=0.0) == 0.0
    assert speed_mps((0.0, 0.0), (100.0, 0.0), dt_s=-1.0) == 0.0


def test_speed_mps_diagonal_motion():
    """Diagonal motion of 100 px (60-80-100) over 1 s at 10 px/m should yield 10 m/s."""
    speed = speed_mps((0.0, 0.0), (60.0, 80.0), dt_s=1.0, pixels_per_meter=10.0)
    assert speed == pytest.approx(10.0)


def test_speed_mps_clamped_to_human_max():
    """Speeds exceeding MAX_HUMAN_SPEED_MPS should be clamped, not returned raw.

    200 px over 1 s at 10 px/m = 20 m/s, which is physically impossible — it
    should be clamped to MAX_HUMAN_SPEED_MPS (10.5 m/s).
    """
    speed = speed_mps((0.0, 0.0), (200.0, 0.0), dt_s=1.0, pixels_per_meter=10.0)
    assert speed == pytest.approx(MAX_HUMAN_SPEED_MPS)


def test_speed_mps_clamp_not_triggered_below_max():
    """A speed below the cap should pass through unchanged."""
    speed = speed_mps((0.0, 0.0), (100.0, 0.0), dt_s=1.0, pixels_per_meter=10.0)
    assert speed == pytest.approx(10.0)
    assert speed < MAX_HUMAN_SPEED_MPS


def test_speed_mps_exactly_at_cap():
    """A speed exactly at the cap should be returned unchanged (min is inclusive)."""
    # 105 px over 1 s at 10 px/m = 10.5 m/s == MAX_HUMAN_SPEED_MPS
    speed = speed_mps((0.0, 0.0), (105.0, 0.0), dt_s=1.0, pixels_per_meter=10.0)
    assert speed == pytest.approx(MAX_HUMAN_SPEED_MPS)


# ── classify_state ─────────────────────────────────────────────────


def test_classify_state_idle():
    """Speeds strictly below 1.2 m/s should classify as idle."""
    assert classify_state(0.0) == "idle"
    assert classify_state(1.19) == "idle"


def test_classify_state_jogging():
    """Speeds in [1.2, 3.5) should classify as jogging."""
    assert classify_state(1.2) == "jogging"
    assert classify_state(2.0) == "jogging"
    assert classify_state(3.49) == "jogging"


def test_classify_state_sprinting():
    """Speeds >= 3.5 m/s should classify as sprinting."""
    assert classify_state(3.5) == "sprinting"
    assert classify_state(5.0) == "sprinting"
    assert classify_state(10.0) == "sprinting"


def test_classify_state_boundary_idle_to_jogging():
    """The exact 1.2 boundary should fall into jogging (lower-inclusive)."""
    assert classify_state(1.2) == "jogging"


def test_classify_state_boundary_jogging_to_sprinting():
    """The exact 3.5 boundary should fall into sprinting (lower-inclusive)."""
    assert classify_state(3.5) == "sprinting"


# ── smooth_positions ───────────────────────────────────────────────


def test_smooth_positions_disabled_window_one():
    """window_size=1 (or 0) should return the positions unchanged."""
    positions = [(0.0, 0.0), (10.0, 5.0), (20.0, 0.0)]
    assert smooth_positions(positions, window_size=1) == positions
    assert smooth_positions(positions, window_size=0) == positions


def test_smooth_positions_empty():
    """An empty trajectory should smooth to an empty list."""
    assert smooth_positions([], window_size=3) == []


def test_smooth_positions_single_point():
    """A single point should be returned unchanged (no neighbors to average)."""
    assert smooth_positions([(7.0, 9.0)], window_size=3) == [(7.0, 9.0)]


def test_smooth_positions_constant_trajectory():
    """A constant trajectory should remain constant after smoothing."""
    positions = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
    smoothed = smooth_positions(positions, window_size=3)
    assert smoothed == [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]


def test_smooth_positions_centered_average_interior():
    """Interior points should be the mean of their centered window.

    For window_size=3, the middle point of [(0,0),(30,0),(60,0)] becomes
    the mean of all three = (30, 0).
    """
    positions = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0)]
    smoothed = smooth_positions(positions, window_size=3)
    assert smoothed[1] == (30.0, 0.0)


def test_smooth_positions_edge_points_use_fewer_samples():
    """Edge points should average only the available samples (clipped window).

    For window_size=3 on [(0,0),(30,0),(60,0)]:
        i=0: window [0,2) -> mean((0,0),(30,0)) = (15, 0)
        i=2: window [1,3) -> mean((30,0),(60,0)) = (45, 0)
    """
    positions = [(0.0, 0.0), (30.0, 0.0), (60.0, 0.0)]
    smoothed = smooth_positions(positions, window_size=3)
    assert smoothed[0] == (15.0, 0.0)
    assert smoothed[2] == (45.0, 0.0)


def test_smooth_positions_reduces_jitter():
    """Smoothing should pull an outlier toward its neighbors.

    A spike at the center of a flat trajectory should be damped toward the
    surrounding values rather than passing through unchanged.
    """
    positions = [(0.0, 0.0), (0.0, 90.0), (0.0, 0.0)]
    smoothed = smooth_positions(positions, window_size=3)
    # Center point becomes mean(0, 90, 0) = 30 — damped from 90 toward 0.
    assert smoothed[1] == (0.0, 30.0)
    # Edges become mean of the two nearest points.
    assert smoothed[0] == (0.0, 45.0)
    assert smoothed[2] == (0.0, 45.0)


def test_smooth_positions_preserves_length():
    """The smoothed trajectory should have the same length as the input."""
    positions = [(float(i), float(i * 2)) for i in range(10)]
    smoothed = smooth_positions(positions, window_size=3)
    assert len(smoothed) == len(positions)


def test_smooth_positions_returns_new_list():
    """smooth_positions should not mutate the input sequence."""
    positions = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    original = list(positions)
    smooth_positions(positions, window_size=3)
    assert list(positions) == original


# ── generate_movement_log ──────────────────────────────────────────


def test_generate_movement_log_dominant_state_jogging():
    """A trajectory of mostly-jogging speeds should report state 'jogging'.

    With pixels_per_meter=10 and dt=1s:
        step 0->1: 20 px  -> 2.0 m/s  (jogging)
        step 1->2: 20 px  -> 2.0 m/s  (jogging)
        step 2->3: 12 px  -> 1.2 m/s  (jogging, boundary)
    Dominant state: jogging. Mean speed: (2.0 + 2.0 + 1.2) / 3.
    Total distance: (20 + 20 + 12) / 10 = 5.2 m.
    """
    positions = [(0.0, 0.0), (20.0, 0.0), (40.0, 0.0), (52.0, 0.0)]
    timestamps = [0.0, 1.0, 2.0, 3.0]
    log = generate_movement_log(
        player_id=7,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
        smoothing_window=1,
    )

    assert isinstance(log, MovementLog)
    assert log.player_id == 7
    assert log.window_start_s == pytest.approx(0.0)
    assert log.window_end_s == pytest.approx(3.0)
    assert log.state == "jogging"
    assert log.avg_speed_mps == pytest.approx((2.0 + 2.0 + 1.2) / 3.0)
    assert log.distance_covered_m == pytest.approx(5.2)
    assert log.juggle_count == 0


def test_generate_movement_log_dominant_state_sprinting():
    """A trajectory of all-sprinting speeds should report state 'sprinting'.

    With pixels_per_meter=10 and dt=1s, each 50 px step -> 5.0 m/s (sprinting).
    """
    positions = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]
    timestamps = [0.0, 1.0, 2.0]
    log = generate_movement_log(
        player_id=3,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
        smoothing_window=1,
    )

    assert log.state == "sprinting"
    assert log.avg_speed_mps == pytest.approx(5.0)
    assert log.distance_covered_m == pytest.approx(10.0)


def test_generate_movement_log_dominant_state_idle():
    """A stationary trajectory should report state 'idle' with zero distance."""
    positions = [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)]
    timestamps = [0.0, 1.0, 2.0]
    log = generate_movement_log(player_id=1, positions=positions, timestamps=timestamps)

    assert log.state == "idle"
    assert log.avg_speed_mps == pytest.approx(0.0)
    assert log.distance_covered_m == pytest.approx(0.0)


def test_generate_movement_log_single_position():
    """A single position (no displacement) should default to idle with zero stats."""
    log = generate_movement_log(
        player_id=42,
        positions=[(5.0, 5.0)],
        timestamps=[1.5],
    )

    assert log.player_id == 42
    assert log.window_start_s == pytest.approx(1.5)
    assert log.window_end_s == pytest.approx(1.5)
    assert log.state == "idle"
    assert log.avg_speed_mps == pytest.approx(0.0)
    assert log.distance_covered_m == pytest.approx(0.0)


def test_generate_movement_log_empty_trajectory():
    """An empty trajectory should produce an idle log with zeroed stats."""
    log = generate_movement_log(player_id=99, positions=[], timestamps=[])

    assert log.player_id == 99
    assert log.window_start_s == pytest.approx(0.0)
    assert log.window_end_s == pytest.approx(0.0)
    assert log.state == "idle"
    assert log.avg_speed_mps == pytest.approx(0.0)
    assert log.distance_covered_m == pytest.approx(0.0)


def test_generate_movement_log_mismatched_lengths_raises():
    """Mismatched positions/timestamps lengths should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        generate_movement_log(
            player_id=1,
            positions=[(0.0, 0.0), (10.0, 0.0)],
            timestamps=[0.0],
        )


def test_generate_movement_log_juggle_count_propagated():
    """The optional juggle_count should be propagated into the MovementLog."""
    positions = [(0.0, 0.0), (50.0, 0.0)]
    timestamps = [0.0, 1.0]
    log = generate_movement_log(
        player_id=11,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
        juggle_count=4,
    )

    assert log.juggle_count == 4


def test_generate_movement_log_default_juggle_count_zero():
    """juggle_count should default to 0 when not provided."""
    positions = [(0.0, 0.0), (10.0, 0.0)]
    timestamps = [0.0, 1.0]
    log = generate_movement_log(player_id=1, positions=positions, timestamps=timestamps)

    assert log.juggle_count == 0


def test_generate_movement_log_uses_default_calibration():
    """When pixels_per_meter is omitted, the module default should be used.

    With DEFAULT_PIXELS_PER_METER and 100 px over 1 s, speed == 10 m/s (sprinting).
    """
    positions = [(0.0, 0.0), (100.0, 0.0)]
    timestamps = [0.0, 1.0]
    log = generate_movement_log(
        player_id=5,
        positions=positions,
        timestamps=timestamps,
        smoothing_window=1,
    )

    assert log.avg_speed_mps == pytest.approx(100.0 / DEFAULT_PIXELS_PER_METER)
    assert log.distance_covered_m == pytest.approx(100.0 / DEFAULT_PIXELS_PER_METER)
    assert log.state == "sprinting"


# ── generate_movement_log with default smoothing ──────────────────


def test_generate_movement_log_applies_default_smoothing():
    """By default, generate_movement_log should smooth positions before computing.

    With pixels_per_meter=10, dt=1s, and window_size=3 on
    [(0,0),(50,0),(100,0)]:
        smoothed = [(25,0),(50,0),(75,0)]
        displacements = 25, 25  -> 2.5 m/s each (jogging)
        distance = 5.0 m, avg_speed = 2.5 m/s, dominant state = jogging.
    """
    positions = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)]
    timestamps = [0.0, 1.0, 2.0]
    log = generate_movement_log(
        player_id=3,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
    )

    assert log.state == "jogging"
    assert log.avg_speed_mps == pytest.approx(2.5)
    assert log.distance_covered_m == pytest.approx(5.0)


def test_generate_movement_log_smoothing_damps_jitter_spike():
    """Smoothing should suppress a single-frame jitter spike in the metrics.

    A trajectory that is otherwise stationary but has one outlier frame
    should report a much lower distance/speed than the raw (unsmoothed) math.
    """
    positions = [(0.0, 0.0), (0.0, 90.0), (0.0, 0.0), (0.0, 0.0)]
    timestamps = [0.0, 1.0, 2.0, 3.0]

    # With smoothing (default window=3): the spike is damped.
    smoothed_log = generate_movement_log(
        player_id=1,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
    )
    # Without smoothing: the raw spike produces a large distance.
    raw_log = generate_movement_log(
        player_id=1,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
        smoothing_window=1,
    )

    assert smoothed_log.distance_covered_m < raw_log.distance_covered_m
    assert smoothed_log.avg_speed_mps < raw_log.avg_speed_mps


def test_generate_movement_log_smoothing_constant_trajectory_unchanged():
    """A constant trajectory should be unaffected by smoothing."""
    positions = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
    timestamps = [0.0, 1.0, 2.0]
    log = generate_movement_log(
        player_id=1,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
    )

    assert log.state == "idle"
    assert log.avg_speed_mps == pytest.approx(0.0)
    assert log.distance_covered_m == pytest.approx(0.0)


def test_generate_movement_log_clamps_unrealistic_speeds():
    """Per-frame speeds above MAX_HUMAN_SPEED_MPS should be clamped in the log.

    With pixels_per_meter=10 and dt=1s, a 200 px jump = 20 m/s raw, which
    should be clamped to 10.5 m/s. With smoothing disabled (window=1) the
    clamp is the only mechanism suppressing the spike.
    """
    positions = [(0.0, 0.0), (200.0, 0.0)]
    timestamps = [0.0, 1.0]
    log = generate_movement_log(
        player_id=9,
        positions=positions,
        timestamps=timestamps,
        pixels_per_meter=10.0,
        smoothing_window=1,
    )

    assert log.avg_speed_mps == pytest.approx(MAX_HUMAN_SPEED_MPS)
    # Distance is computed from raw pixel displacement (not clamped), so it
    # still reflects the actual 20 m traveled — only the *speed* is clamped.
    assert log.distance_covered_m == pytest.approx(20.0)
    assert log.state == "sprinting"


def test_movement_log_schema_defaults_and_validation():
    """MovementLog should validate with all required fields and default juggle_count=0."""
    log = MovementLog(
        player_id=1,
        window_start_s=0.0,
        window_end_s=10.0,
        state="jogging",
        avg_speed_mps=2.5,
        distance_covered_m=25.0,
    )
    assert log.juggle_count == 0
    assert log.state == "jogging"


def test_movement_log_schema_rejects_negative_speed():
    """MovementLog should reject a negative avg_speed_mps (ge=0.0 constraint)."""
    with pytest.raises(ValidationError):
        MovementLog(
            player_id=1,
            window_start_s=0.0,
            window_end_s=10.0,
            state="idle",
            avg_speed_mps=-1.0,
            distance_covered_m=0.0,
        )
