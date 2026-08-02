"""Unit tests for player Re-Identification — feature extraction, gallery, and matching."""

import numpy as np
import pytest

from vision.reid import (
    DEFAULT_HISTORY_LEN,
    DEFAULT_LOST_TTL_FRAMES,
    FeatureExtractor,
    ReIDGallery,
    _l2_normalize,
    cosine_similarity,
)

# ── helpers / fixtures ─────────────────────────────────────────────


def _unit(dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic L2-normalized vector of length ``dim``."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return _l2_normalize(v)


def _mock_backbone(dim: int = 8):
    """Return a numpy backbone + preprocess pair that yields a deterministic embedding.

    The embedding is derived from the crop's mean intensity so distinct crops
    produce distinct (and reproducible) vectors without needing torch.
    """

    def preprocess(crop):
        return np.asarray(crop, dtype=np.float32)

    def backbone(x):
        # Reduce any-shape input to a fixed-dim vector seeded by the input mean.
        scalar = float(np.mean(x)) if x.size > 0 else 0.0
        rng = np.random.default_rng(int(abs(scalar) * 1000) % (2**31))
        return rng.standard_normal(dim).astype(np.float32)

    return backbone, preprocess


def _constant_backbone(vector: np.ndarray):
    """Backbone/preprocess pair that always returns ``vector`` (ignores input)."""

    def preprocess(crop):
        return np.asarray(crop, dtype=np.float32)

    def backbone(_x):
        return np.asarray(vector, dtype=np.float32)

    return backbone, preprocess


def _dummy_crop(value: int = 128, size: tuple = (32, 32, 3)) -> np.ndarray:
    """A constant-value uint8 image crop."""
    return np.full(size, value, dtype=np.uint8)


# ── _l2_normalize ──────────────────────────────────────────────────


def test_l2_normalize_unit_length():
    """A normalized vector must have L2 norm 1."""
    v = _l2_normalize(np.array([3.0, 4.0]))
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert v == pytest.approx(np.array([0.6, 0.8]))


def test_l2_normalize_zero_vector_is_zero():
    """A zero vector should stay zero (no NaN)."""
    v = _l2_normalize(np.zeros(4))
    assert np.linalg.norm(v) == pytest.approx(0.0)
    assert not np.any(np.isnan(v))


def test_l2_normalize_returns_1d_float32():
    """Output should be a 1D float32 array."""
    v = _l2_normalize(np.array([[1.0, 2.0, 2.0]]))
    assert v.ndim == 1
    assert v.dtype == np.float32


# ── cosine_similarity ──────────────────────────────────────────────


def test_cosine_similarity_identical_vectors():
    """Identical unit vectors have cosine similarity 1.0."""
    v = _unit(5, seed=1)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors have cosine similarity 0.0."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors have cosine similarity -1.0."""
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    """A zero-norm operand should yield 0.0 (no spurious match)."""
    assert cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0
    assert cosine_similarity(np.array([1.0, 0.0, 0.0]), np.zeros(3)) == 0.0


def test_cosine_similarity_scale_invariant():
    """Cosine similarity is invariant to vector magnitude."""
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, 5 * a) == pytest.approx(1.0)


# ── FeatureExtractor (mock backbone — no torch required) ───────────


def test_feature_extractor_normalizes_to_unit_length():
    """extract() must return an L2-normalized vector."""
    backbone, preprocess = _constant_backbone(np.array([3.0, 4.0]))
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    emb = extractor.extract(_dummy_crop())
    assert np.linalg.norm(emb) == pytest.approx(1.0)
    assert emb == pytest.approx(np.array([0.6, 0.8]))


def test_feature_extractor_zero_vector_stays_zero():
    """A backbone returning zero should yield a zero embedding (no NaN)."""
    backbone, preprocess = _constant_backbone(np.zeros(4))
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    emb = extractor.extract(_dummy_crop())
    assert np.linalg.norm(emb) == pytest.approx(0.0)
    assert not np.any(np.isnan(emb))


def test_feature_extractor_is_1d_float32():
    """The embedding should be a 1D float32 array."""
    backbone, preprocess = _mock_backbone(dim=8)
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    emb = extractor.extract(_dummy_crop())
    assert emb.ndim == 1
    assert emb.dtype == np.float32
    assert emb.shape[0] == 8


def test_feature_extractor_deterministic_for_same_crop():
    """The same crop should yield the same embedding (mock is deterministic)."""
    backbone, preprocess = _mock_backbone(dim=8)
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    crop = _dummy_crop(value=100)
    e1 = extractor.extract(crop)
    e2 = extractor.extract(crop)
    assert np.allclose(e1, e2)


def test_feature_extractor_batch_matches_single():
    """extract_batch results should match per-crop extract() results."""
    backbone, preprocess = _mock_backbone(dim=6)
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    crops = [_dummy_crop(value=v) for v in (10, 50, 200)]
    batch = extractor.extract_batch(crops)
    singles = np.stack([extractor.extract(c) for c in crops], axis=0)
    assert batch.shape == (3, 6)
    assert np.allclose(batch, singles)


def test_feature_extractor_empty_batch_shape():
    """An empty batch should return a (0, D) array without error."""
    backbone, preprocess = _mock_backbone(dim=4)
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    # Populate embedding_dim via a single extract first.
    extractor.extract(_dummy_crop())
    batch = extractor.extract_batch([])
    assert batch.shape == (0, extractor.embedding_dim)


def test_feature_extractor_embedding_dim_populated():
    """embedding_dim should be set after the first extract call."""
    backbone, preprocess = _mock_backbone(dim=10)
    extractor = FeatureExtractor(backbone=backbone, preprocess=preprocess)
    assert extractor.embedding_dim is None
    extractor.extract(_dummy_crop())
    assert extractor.embedding_dim == 10


def test_feature_extractor_default_preprocess_fallback_flatten():
    """An injected backbone without preprocess should use the flatten fallback."""
    backbone, _ = _mock_backbone(dim=4)
    extractor = FeatureExtractor(backbone=backbone)  # no preprocess -> default
    emb = extractor.extract(_dummy_crop(value=200, size=(4, 4, 3)))
    assert np.linalg.norm(emb) == pytest.approx(1.0)


def test_feature_extractor_default_backbone_requires_torch():
    """The default MobileNetV3 backbone needs torch/torchvision; skip if absent."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    extractor = FeatureExtractor()
    emb = extractor.extract(_dummy_crop())
    assert np.linalg.norm(emb) == pytest.approx(1.0)
    assert emb.ndim == 1


# ── ReIDGallery — register / active / lost ─────────────────────────


def test_gallery_register_marks_active():
    """register() should add the ID to active and store the embedding."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1))
    assert 1 in gallery.active_ids
    assert 1 not in gallery.lost_ids
    assert gallery.history_length(1) == 1


def test_gallery_register_renormalize_embedding():
    """register() should store an L2-normalized embedding even if input is raw."""
    gallery = ReIDGallery()
    gallery.register(2, np.array([3.0, 4.0]))
    rep = gallery.representative(2)
    assert np.linalg.norm(rep) == pytest.approx(1.0)


def test_gallery_mark_lost_moves_to_lost_pool():
    """mark_lost() should move an ID from active to lost."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1))
    assert 1 in gallery.active_ids
    gallery.mark_lost(1, frame_idx=10)
    assert 1 not in gallery.active_ids
    assert 1 in gallery.lost_ids


def test_gallery_register_reactivates_lost_id():
    """Re-registering a lost ID should move it back to active."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1))
    gallery.mark_lost(1, frame_idx=5)
    assert 1 in gallery.lost_ids
    gallery.register(1, _unit(8, seed=2), frame_idx=6)
    assert 1 in gallery.active_ids
    assert 1 not in gallery.lost_ids


def test_gallery_history_rolling_window_maxlen():
    """History should be capped at history_len (rolling window)."""
    gallery = ReIDGallery(history_len=3)
    for i in range(5):
        gallery.register(1, _unit(8, seed=i))
    assert gallery.history_length(1) == 3


def test_gallery_history_default_maxlen():
    """Default history length should be DEFAULT_HISTORY_LEN."""
    gallery = ReIDGallery()
    for i in range(DEFAULT_HISTORY_LEN + 5):
        gallery.register(1, _unit(8, seed=i))
    assert gallery.history_length(1) == DEFAULT_HISTORY_LEN


def test_gallery_representative_is_mean_and_normalized():
    """representative() should return the L2-normalized mean of stored embeddings."""
    gallery = ReIDGallery()
    gallery.register(1, np.array([1.0, 0.0]))
    gallery.register(1, np.array([0.0, 1.0]))
    rep = gallery.representative(1)
    # mean = [0.5, 0.5] -> normalized = [~0.707, ~0.707]
    assert np.linalg.norm(rep) == pytest.approx(1.0)
    assert rep == pytest.approx(np.array([1.0, 1.0]) / np.sqrt(2.0))


def test_gallery_representative_unknown_id_is_none():
    """representative() for an unknown ID should return None."""
    gallery = ReIDGallery()
    assert gallery.representative(999) is None


def test_gallery_representative_empty_history_is_none():
    """representative() for an ID with no embeddings should return None."""
    gallery = ReIDGallery()
    gallery.mark_lost(1, frame_idx=0)  # lost but no history
    assert gallery.representative(1) is None


# ── ReIDGallery — find_best_match ──────────────────────────────────


def test_find_best_match_above_threshold():
    """A query close to a lost track's embedding should match it."""
    gallery = ReIDGallery()
    target = _unit(8, seed=42)
    gallery.register(7, target, frame_idx=0)
    gallery.mark_lost(7, frame_idx=1)
    # Query is the same direction -> cosine sim ~1.0 >= threshold.
    match = gallery.find_best_match(target, threshold=0.80)
    assert match == 7


def test_find_best_match_below_threshold_returns_none():
    """A query dissimilar to all lost tracks should return None."""
    gallery = ReIDGallery()
    gallery.register(7, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(7, frame_idx=1)
    query = _unit(8, seed=99)  # very different direction
    match = gallery.find_best_match(query, threshold=0.99)
    assert match is None


def test_find_best_match_empty_gallery_returns_none():
    """An empty gallery should never match."""
    gallery = ReIDGallery()
    assert gallery.find_best_match(_unit(8, seed=0)) is None


def test_find_best_match_ignores_active_by_default():
    """By default, active tracks should not be matched (only lost ones)."""
    gallery = ReIDGallery()
    target = _unit(8, seed=7)
    gallery.register(5, target, frame_idx=0)  # active, not lost
    assert gallery.find_best_match(target, threshold=0.80) is None


def test_find_best_match_includes_active_when_requested():
    """search_active=True should also match active tracks."""
    gallery = ReIDGallery()
    target = _unit(8, seed=7)
    gallery.register(5, target, frame_idx=0)
    match = gallery.find_best_match(target, threshold=0.80, search_active=True)
    assert match == 5


def test_find_best_match_picks_highest_similarity():
    """With multiple lost candidates, the closest one should win."""
    gallery = ReIDGallery()
    a = _unit(8, seed=1)
    b = _unit(8, seed=2)
    query = a + 0.01 * b  # much closer to a
    gallery.register(10, a, frame_idx=0)
    gallery.register(20, b, frame_idx=0)
    gallery.mark_lost(10, frame_idx=1)
    gallery.mark_lost(20, frame_idx=1)
    match = gallery.find_best_match(_l2_normalize(query), threshold=0.5)
    assert match == 10


def test_find_best_match_reactivates_on_register():
    """After a match is found and re-registered, the ID should be active again."""
    gallery = ReIDGallery()
    target = _unit(8, seed=3)
    gallery.register(9, target, frame_idx=0)
    gallery.mark_lost(9, frame_idx=1)
    match = gallery.find_best_match(target, threshold=0.80)
    assert match == 9
    gallery.register(match, target, frame_idx=2)
    assert 9 in gallery.active_ids
    assert 9 not in gallery.lost_ids


def test_best_similarity_returns_score():
    """best_similarity() should return the raw score without thresholding."""
    gallery = ReIDGallery()
    target = _unit(8, seed=5)
    gallery.register(3, target, frame_idx=0)
    gallery.mark_lost(3, frame_idx=1)
    pid, sim = gallery.best_similarity(target)
    assert pid == 3
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_best_similarity_empty_gallery():
    """best_similarity() on an empty gallery should return (None, -1.0)."""
    gallery = ReIDGallery()
    pid, sim = gallery.best_similarity(_unit(8, seed=0))
    assert pid is None
    assert sim == pytest.approx(-1.0)


# ── ReIDGallery — eviction ─────────────────────────────────────────


def test_evict_removes_old_lost_tracks():
    """Lost tracks older than lost_ttl_frames should be evicted."""
    gallery = ReIDGallery(lost_ttl_frames=10)
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(1, frame_idx=5)
    evicted = gallery.evict(current_frame=20)  # 20 - 5 = 15 > 10
    assert 1 in evicted
    assert 1 not in gallery.lost_ids
    assert 1 not in gallery.known_ids


def test_evict_keeps_recent_lost_tracks():
    """Lost tracks within the TTL window should be retained."""
    gallery = ReIDGallery(lost_ttl_frames=10)
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(1, frame_idx=15)
    evicted = gallery.evict(current_frame=20)  # 20 - 15 = 5 <= 10
    assert evicted == []
    assert 1 in gallery.lost_ids


def test_evict_boundary_exactly_at_ttl():
    """A lost track exactly at TTL (diff == ttl) should be retained (strict >)."""
    gallery = ReIDGallery(lost_ttl_frames=10)
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(1, frame_idx=10)
    evicted = gallery.evict(current_frame=20)  # 20 - 10 = 10, not > 10
    assert evicted == []
    assert 1 in gallery.lost_ids


def test_evict_does_not_remove_active_tracks():
    """Active tracks should never be evicted (they're not in the lost pool)."""
    gallery = ReIDGallery(lost_ttl_frames=5)
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    evicted = gallery.evict(current_frame=1000)
    assert evicted == []
    assert 1 in gallery.active_ids


def test_evict_default_ttl():
    """Default TTL should be DEFAULT_LOST_TTL_FRAMES."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(1, frame_idx=0)
    # Just under TTL -> retained.
    assert gallery.evict(current_frame=DEFAULT_LOST_TTL_FRAMES) == []
    # Over TTL -> evicted.
    assert gallery.evict(current_frame=DEFAULT_LOST_TTL_FRAMES + 1) == [1]


# ── ReIDGallery — remove / known_ids ───────────────────────────────


def test_remove_fully_clears_id():
    """remove() should clear an ID from history, active, and lost pools."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.mark_lost(1, frame_idx=1)
    gallery.remove(1)
    assert 1 not in gallery.known_ids
    assert 1 not in gallery.active_ids
    assert 1 not in gallery.lost_ids


def test_known_ids_lists_all_with_history():
    """known_ids should include every ID that has stored embeddings."""
    gallery = ReIDGallery()
    gallery.register(1, _unit(8, seed=1), frame_idx=0)
    gallery.register(2, _unit(8, seed=2), frame_idx=0)
    gallery.mark_lost(1, frame_idx=1)
    assert gallery.known_ids == {1, 2}


# ── ReIDGallery — end-to-end ReID scenario ─────────────────────────


def test_reid_scenario_lost_then_rematched():
    """End-to-end: a player is lost then re-appears with a new ID and is rematched.

    Simulates the core ReID use case: player 5 is tracked, gets occluded
    (marked lost), then re-appears. A query embedding close to the stored
    history should re-match to canonical ID 5.
    """
    gallery = ReIDGallery()
    player_embedding = _unit(16, seed=123)

    # Track player 5 for several frames.
    for i in range(5):
        gallery.register(5, player_embedding + 1e-3 * _unit(16, seed=i), frame_idx=i)

    # Occlusion — player 5 disappears.
    gallery.mark_lost(5, frame_idx=5)

    # Some frames pass (within TTL).
    gallery.evict(current_frame=20)

    # Player re-appears with a slightly noisy version of the same embedding.
    query = player_embedding + 1e-2 * _unit(16, seed=999)
    match = gallery.find_best_match(_l2_normalize(query), threshold=0.80)
    assert match == 5

    # Re-register under the matched canonical ID.
    gallery.register(match, _l2_normalize(query), frame_idx=20)
    assert 5 in gallery.active_ids
    assert 5 not in gallery.lost_ids


def test_reid_scenario_two_distinct_players_not_confused():
    """Two players with distinct embeddings should not be cross-matched."""
    gallery = ReIDGallery()
    p1 = _unit(16, seed=1)
    p2 = _unit(16, seed=2)

    gallery.register(1, p1, frame_idx=0)
    gallery.register(2, p2, frame_idx=0)
    gallery.mark_lost(1, frame_idx=1)
    gallery.mark_lost(2, frame_idx=1)

    # Query close to p2 should match 2, not 1.
    query = _l2_normalize(p2 + 1e-3 * _unit(16, seed=50))
    assert gallery.find_best_match(query, threshold=0.80) == 2
