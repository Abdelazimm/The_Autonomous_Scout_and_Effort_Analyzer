"""Player Re-Identification via visual embeddings.

Maintains persistent player IDs across brief occlusions and tracker resets by
comparing L2-normalized feature embeddings of player crops against a rolling
gallery of recently seen / recently lost tracks. Uses cosine similarity and a
configurable threshold to re-map transient ByteTrack IDs to canonical IDs.

The default ``FeatureExtractor`` uses a pretrained ``torchvision`` MobileNetV3
backbone (lazy-imported so this module imports without torch installed). A
custom backbone / preprocess callable can be injected for testing or to swap
in a dedicated ReID model.
"""

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default ReID hyperparameters
DEFAULT_HISTORY_LEN = 30
DEFAULT_LOST_TTL_FRAMES = 90
DEFAULT_REID_THRESHOLD = 0.80
DEFAULT_INPUT_SIZE: Tuple[int, int] = (224, 224)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized copy of ``vector`` (1D float32). Zero-safe."""
    v = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return v
    return v / norm


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, in ``[-1, 1]``.

    Returns ``0.0`` if either vector has zero norm (so a degenerate query or
    gallery entry never produces a spurious match).
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a_arr))
    nb = float(np.linalg.norm(b_arr))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (na * nb))


class FeatureExtractor:
    """Extract L2-normalized 1D feature embeddings from image crops.

    By default a pretrained ``torchvision.models.mobilenet_v3_small`` backbone
    is used (classifier stripped to ``Identity`` so the pooled 576-d feature is
    returned). Torch / torchvision are imported lazily on first use.

    Args:
        backbone: Optional callable ``tensor -> features``. If ``None``, the
            default MobileNetV3 backbone is loaded. Injecting a mock callable
            (numpy or torch) avoids downloading weights — used in tests.
        preprocess: Optional callable ``crop -> tensor`` applied before the
            backbone. If ``None``, a standard ImageNet-style transform is
            built for the default backbone (or a minimal flatten transform for
            an injected backbone).
        device: Torch device for the default backbone (e.g. ``"cpu"`` or
            ``"cuda"``). Ignored when ``backbone`` is injected.
        input_size: Resize target ``(H, W)`` for the default preprocess.
    """

    def __init__(
        self,
        backbone: Optional[Callable] = None,
        preprocess: Optional[Callable] = None,
        device: str = "cpu",
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        self._device = device
        self._input_size = input_size
        self._embedding_dim: Optional[int] = None

        if backbone is not None:
            self._backbone = backbone
            self._preprocess = preprocess if preprocess is not None else self._default_preprocess
        else:
            self._backbone, default_preprocess = self._load_default_backbone()
            self._preprocess = preprocess if preprocess is not None else default_preprocess

    @staticmethod
    def _default_preprocess(crop: np.ndarray) -> np.ndarray:
        """Minimal fallback preprocess for an injected (non-torch) backbone."""
        img = np.asarray(crop)
        if img.ndim == 3:  # HWC -> flatten
            return img.astype(np.float32).ravel()
        return img.astype(np.float32).ravel()

    def _load_default_backbone(self) -> Tuple[Callable, Callable]:
        """Lazy-load the default MobileNetV3 backbone + preprocess transform."""
        import torch
        from torch import nn
        from torchvision import models, transforms

        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        # Strip the classifier — forward() then returns the 576-d pooled feature.
        model.classifier = nn.Identity()
        model.eval()
        model.to(self._device)

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self._input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        def backbone_fn(x):
            with torch.no_grad():
                x = x.to(self._device)
                feats = model(x)
                if feats.ndim > 2:  # (N, C, 1, 1) -> (N, C)
                    feats = feats.mean(dim=(2, 3))
                return feats

        def preprocess_fn(crop: np.ndarray):
            # crop: HxWx3 uint8 (BGR or RGB). ToPILImage expects HWC uint8.
            img = np.asarray(crop)
            if img.ndim == 2:  # grayscale -> 3-channel
                img = np.stack([img] * 3, axis=-1)
            tensor = preprocess(img)
            return tensor.unsqueeze(0)  # add batch dim

        return backbone_fn, preprocess_fn

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized 1D embedding from a single crop.

        Args:
            crop: ``HxWx3`` uint8 image crop (numpy array).

        Returns:
            1D float32 numpy array with unit L2 norm.
        """
        x = self._preprocess(crop)
        feats = self._backbone(x)
        arr = self._to_numpy_vector(feats)
        emb = _l2_normalize(arr)
        if self._embedding_dim is None:
            self._embedding_dim = emb.shape[0]
        return emb

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized embeddings for a batch of crops.

        Args:
            crops: Sequence of ``HxWx3`` uint8 image crops.

        Returns:
            ``(N, D)`` float32 numpy array with each row L2-normalized.
        """
        if not crops:
            dim = self._embedding_dim if self._embedding_dim is not None else 0
            return np.empty((0, dim), dtype=np.float32)
        return np.stack([self.extract(c) for c in crops], axis=0)

    @staticmethod
    def _to_numpy_vector(feats) -> np.ndarray:
        """Convert a backbone output (torch tensor or numpy array) to 1D numpy."""
        if hasattr(feats, "detach"):  # torch tensor
            feats = feats.detach().cpu().numpy()
        arr = np.asarray(feats, dtype=np.float32)
        return arr.ravel()

    @property
    def embedding_dim(self) -> Optional[int]:
        """Dimensionality of the embedding, or ``None`` until first extract."""
        return self._embedding_dim


class ReIDGallery:
    """Rolling gallery of player embeddings for ReID matching.

    Keeps a bounded rolling history (default last 30 frames) of embeddings per
    canonical player ID. Active IDs are currently tracked; lost IDs have
    disappeared from the tracker but are retained for ``lost_ttl_frames`` so a
    re-appearing track can be re-matched to its canonical ID via
    ``find_best_match``.

    Args:
        history_len: Max embeddings retained per player ID.
        lost_ttl_frames: Frames a lost ID is kept before eviction.
    """

    def __init__(
        self,
        history_len: int = DEFAULT_HISTORY_LEN,
        lost_ttl_frames: int = DEFAULT_LOST_TTL_FRAMES,
    ) -> None:
        self._history_len = history_len
        self._lost_ttl_frames = lost_ttl_frames
        self._history: Dict[int, Deque[np.ndarray]] = {}
        self._active: set = set()
        self._lost: Dict[int, int] = {}  # player_id -> last_frame_seen

    def register(self, player_id: int, embedding: Sequence[float], frame_idx: Optional[int] = None) -> None:
        """Add an embedding to ``player_id``'s rolling history and mark it active."""
        if player_id not in self._history:
            self._history[player_id] = deque(maxlen=self._history_len)
        self._history[player_id].append(_l2_normalize(embedding))
        self._active.add(player_id)
        self._lost.pop(player_id, None)

    def mark_lost(self, player_id: int, frame_idx: Optional[int] = None) -> None:
        """Move ``player_id`` from active to lost (retained for matching/eviction)."""
        self._active.discard(player_id)
        self._lost[player_id] = frame_idx if frame_idx is not None else 0

    def representative(self, player_id: int) -> Optional[np.ndarray]:
        """Return the L2-normalized mean embedding for ``player_id``, or ``None``."""
        embs = list(self._history.get(player_id, []))
        if not embs:
            return None
        mean = np.mean(np.stack(embs), axis=0)
        return _l2_normalize(mean)

    def find_best_match(
        self,
        query_embedding: Sequence[float],
        threshold: float = DEFAULT_REID_THRESHOLD,
        search_active: bool = False,
    ) -> Optional[int]:
        """Find the best-matching player ID for ``query_embedding``.

        Searches the lost pool by default (recently lost tracks). Set
        ``search_active=True`` to also compare against active tracks.

        Args:
            query_embedding: L2-normalized (or raw) embedding of the query crop.
            threshold: Minimum cosine similarity to accept a match.
            search_active: If ``True``, include active IDs in the search.

        Returns:
            The best-matching canonical player ID, or ``None`` if no candidate
            reaches ``threshold``.
        """
        query = _l2_normalize(query_embedding)
        candidate_ids = list(self._lost.keys())
        if search_active:
            candidate_ids += [pid for pid in self._history if pid not in self._lost]

        best_id: Optional[int] = None
        best_sim = -1.0
        for pid in candidate_ids:
            rep = self.representative(pid)
            if rep is None:
                continue
            sim = cosine_similarity(query, rep)
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_id is not None and best_sim >= threshold:
            return best_id
        return None

    def best_similarity(
        self,
        query_embedding: Sequence[float],
        search_active: bool = False,
    ) -> Tuple[Optional[int], float]:
        """Return ``(best_id, best_similarity)`` without applying a threshold."""
        query = _l2_normalize(query_embedding)
        candidate_ids = list(self._lost.keys())
        if search_active:
            candidate_ids += [pid for pid in self._history if pid not in self._lost]

        best_id: Optional[int] = None
        best_sim = -1.0
        for pid in candidate_ids:
            rep = self.representative(pid)
            if rep is None:
                continue
            sim = cosine_similarity(query, rep)
            if sim > best_sim:
                best_sim = sim
                best_id = pid
        return best_id, best_sim

    def evict(self, current_frame: int) -> List[int]:
        """Remove lost IDs older than ``lost_ttl_frames``. Returns evicted IDs."""
        expired = [
            pid for pid, last_seen in self._lost.items()
            if current_frame - last_seen > self._lost_ttl_frames
        ]
        for pid in expired:
            self._history.pop(pid, None)
            self._lost.pop(pid, None)
        return expired

    def remove(self, player_id: int) -> None:
        """Fully remove a player ID from history, active, and lost pools."""
        self._history.pop(player_id, None)
        self._active.discard(player_id)
        self._lost.pop(player_id, None)

    @property
    def active_ids(self) -> set:
        """Set of currently-active canonical player IDs."""
        return set(self._active)

    @property
    def lost_ids(self) -> set:
        """Set of lost (retained) canonical player IDs."""
        return set(self._lost.keys())

    @property
    def known_ids(self) -> set:
        """Set of all player IDs with stored history."""
        return set(self._history.keys())

    def history_length(self, player_id: int) -> int:
        """Number of embeddings currently stored for ``player_id``."""
        return len(self._history.get(player_id, []))
