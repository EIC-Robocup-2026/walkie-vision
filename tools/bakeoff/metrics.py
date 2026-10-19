"""Retrieval and separation metrics for the embedding bake-off.

Every function here is pure: arrays and numbers in, numbers out, with no
model, file or random state. Embeddings must have unit length, so that a dot
product is a cosine similarity. Scores are compared in float64.

Three levels of text-to-crop retrieval are scored (see docs/guides/bakeoff.md):

- object_mean: each object is one vector, the mean of its crop vectors
  scaled back to length 1. This is how walkie-database's object memory
  stores an object.
- object_max: each object scores the best of its crops.
- crop: every crop is ranked on its own, and a hit is any crop of a target
  object.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

LEVELS = ("object_mean", "object_max", "crop")
KS = (1, 5)
TIE_TOL = 1e-6
"""Scores closer than this count as a tie."""


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, in float64."""
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalise a zero vector")
    return x / norms


def hit_probability(n_above: int, n_tied_targets: int, n_tied_others: int, k: int) -> float:
    """Return the chance that a target is in the top k when ties are ordered at random.

    n_above: non-targets that score strictly higher than the best target.
    n_tied_targets: targets that score the same as the best target (at least 1).
    n_tied_others: non-targets that score the same as the best target.

    The tied candidates fill the r = k - n_above places left in the top k. A
    miss means all r places go to tied non-targets, which has probability
    C(n_tied_others, r) / C(n_tied_targets + n_tied_others, r).
    """
    if n_tied_targets < 1:
        raise ValueError("n_tied_targets must be at least 1: the best target ties with itself")
    if min(n_above, n_tied_others) < 0 or k < 1:
        raise ValueError("counts must not be negative and k must be at least 1")
    r = k - n_above
    if r <= 0:
        return 0.0
    if r > n_tied_others:
        return 1.0
    return 1.0 - math.comb(n_tied_others, r) / math.comb(n_tied_targets + n_tied_others, r)


@dataclass(frozen=True)
class Ranked:
    """Where the targets of one query landed among the candidates."""

    rank: int
    """1 + the non-targets that score strictly higher than the best target."""
    n_tied: int
    """Non-targets that score the same as the best target."""
    hits: dict[int, float]
    """k -> the chance that a target is in the top k (0 or 1 without ties)."""
    top: str
    """The highest-scoring candidate (the first one, on a tie)."""
    top_score: float
    best_target: str
    best_target_score: float


def rank_targets(
    scores: Sequence[float],
    candidates: Sequence[str],
    targets: Collection[str],
    ks: Sequence[int] = KS,
    tol: float = TIE_TOL,
) -> Ranked:
    """Rank one query's candidates and score the best-placed target.

    candidates[i] is the object id of the candidate with scores[i]. Several
    candidates may share an object id (crop level), and any candidate whose
    object id is in targets is a target.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(candidates):
        raise ValueError("scores and candidates must have the same length")
    is_target = np.array([c in targets for c in candidates], dtype=bool)
    if not is_target.any():
        raise ValueError("no candidate is a target")
    best = float(scores[is_target].max())
    others = scores[~is_target]
    n_above = int(np.count_nonzero(others > best + tol))
    n_tied = int(np.count_nonzero(np.abs(others - best) <= tol))
    n_tied_targets = int(np.count_nonzero(scores[is_target] >= best - tol))
    top = int(np.argmax(scores))
    best_index = int(np.flatnonzero(is_target & (scores == best))[0])
    return Ranked(
        rank=1 + n_above,
        n_tied=n_tied,
        hits={k: hit_probability(n_above, n_tied_targets, n_tied, k) for k in ks},
        top=candidates[top],
        top_score=float(scores[top]),
        best_target=candidates[best_index],
        best_target_score=best,
    )


def recall_at_k(ranked: Sequence[Ranked], k: int) -> float:
    """Mean of the per-query hit chances at k; NaN when there are no queries."""
    if not ranked:
        return math.nan
    return float(np.mean([r.hits[k] for r in ranked]))


def object_order(crop_objects: Sequence[str]) -> list[str]:
    """Object ids in the order their first crop appears."""
    return list(dict.fromkeys(crop_objects))


def object_vectors(crop_emb: np.ndarray, crop_objects: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Return one vector per object: the mean of its crop vectors, scaled to length 1."""
    crop_emb = np.asarray(crop_emb, dtype=np.float64)
    objects = object_order(crop_objects)
    labels = np.asarray(crop_objects)
    vectors = np.stack([crop_emb[labels == o].mean(axis=0) for o in objects])
    return normalize(vectors), objects


def level_scores(
    level: str, text_emb: np.ndarray, crop_emb: np.ndarray, crop_objects: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return the query x candidate scores for one level, and each candidate's object id."""
    text_emb = np.asarray(text_emb, dtype=np.float64)
    crop_emb = np.asarray(crop_emb, dtype=np.float64)
    if len(crop_emb) != len(crop_objects):
        raise ValueError("crop_emb and crop_objects must have the same length")
    if level == "crop":
        return text_emb @ crop_emb.T, list(crop_objects)
    if level == "object_mean":
        vectors, objects = object_vectors(crop_emb, crop_objects)
        return text_emb @ vectors.T, objects
    if level == "object_max":
        sims = text_emb @ crop_emb.T
        objects = object_order(crop_objects)
        labels = np.asarray(crop_objects)
        return np.stack([sims[:, labels == o].max(axis=1) for o in objects], axis=1), objects
    raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")


@dataclass(frozen=True)
class Separation:
    """How well a model tells two similar objects apart, from their crops alone."""

    n_a: int
    n_b: int
    within: float
    """Mean cosine over every pair of two different crops of the same object, both objects pooled."""
    between: float
    """Mean cosine over every (crop of A, crop of B) pair."""
    margin: float
    """within - between. Higher is better; 0 or less means the views of one
    object are no closer to each other than to the other object."""
    nn_accuracy: float
    """Leave-one-out: the share of crops that are closer to the mean of their
    own object's other crops than to the mean of the other object's crops.
    A tie counts as half. Only crops whose object has another crop count."""


def pair_separation(emb_a: np.ndarray, emb_b: np.ndarray, tol: float = TIE_TOL) -> Separation:
    """Compare the crops of object A with the crops of object B.

    Values that need two crops of one object are NaN when neither object
    has two.
    """
    a = np.asarray(emb_a, dtype=np.float64)
    b = np.asarray(emb_b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("each object needs at least one crop")
    same = [float(x @ y) for group in (a, b) for x, y in combinations(group, 2)]
    within = float(np.mean(same)) if same else math.nan
    between = float((a @ b.T).mean())
    votes = []
    for own, other in ((a, b), (b, a)):
        if len(own) < 2:
            continue
        other_mean = normalize(other.mean(axis=0))
        for i in range(len(own)):
            rest = normalize(np.delete(own, i, axis=0).mean(axis=0))
            d = float(own[i] @ rest) - float(own[i] @ other_mean)
            votes.append(1.0 if d > tol else 0.5 if d >= -tol else 0.0)
    return Separation(
        n_a=len(a),
        n_b=len(b),
        within=within,
        between=between,
        margin=within - between,
        nn_accuracy=float(np.mean(votes)) if votes else math.nan,
    )


# The decision rule. The constants were set before any bake-off data was
# seen; docs/guides/bakeoff.md explains each one. Change them only with a
# reason that does not come from the results they decide.
BASELINE = "clip_vit_b16"
DEFAULT = "siglip2_b16_256"
CHALLENGERS = ("eg2_t70", "eg2_t140")
"""Lower soft-token budget first: on a tie the cheaper one wins."""


@dataclass(frozen=True)
class Rule:
    level: str = "object_mean"
    buckets: tuple[int, ...] = (32, 48, 64)
    min_gain_r5: float = 0.05
    """A challenger must beat SigLIP 2's R@5 by at least this much."""
    max_image_ms: float = 10.0
    """Batched image embedding time per crop, on the robot."""
    max_text_ms: float = 50.0
    """One query embedded on its own, on the robot."""
    max_vram_mib: float = 3072.0
    """Peak memory PyTorch allocated while the model ran."""


@dataclass(frozen=True)
class ModelScore:
    """One model's results, as far as the decision rule needs them."""

    r1: float
    """Mean R@1 over the rule's buckets."""
    r5: float
    """Mean R@5 over the rule's buckets."""
    hits5: tuple[float, ...]
    """Hit@5 per (bucket, query), in the same order for every model."""
    image_ms: float | None
    text_ms: float | None
    vram_mib: float | None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    verdict: str
    """A model name, "investigate", or "incomplete"."""
    reasons: tuple[str, ...]


def _within(value: float | None, limit: float) -> bool:
    return value is not None and not math.isnan(value) and value <= limit


def decide(
    scores: Mapping[str, ModelScore],
    rule: Rule = Rule(),
    baseline: str = BASELINE,
    default: str = DEFAULT,
    challengers: Sequence[str] = CHALLENGERS,
) -> Decision:
    """Pick the model for walkie-database's object and room vectors.

    SigLIP 2 wins unless an EmbeddingGemma 2 budget beats it on R@5 by
    rule.min_gain_r5, is not worse on R@1, wins more queries than it loses
    at hit@5, and fits every cost limit. Among challengers that qualify, the
    highest R@5 wins, and a tie goes to the lower budget. If SigLIP 2 scores
    below CLIP, something is wrong with the data or the setup, and no model
    is picked.
    """
    eps = 1e-9
    if default not in scores or math.isnan(scores[default].r5):
        return Decision("incomplete", (f"{default} has no R@5 on buckets {rule.buckets}",))
    d = scores[default]
    reasons = [f"{default}: R@5 {d.r5:.3f}, R@1 {d.r1:.3f}"]
    if baseline in scores and not math.isnan(scores[baseline].r5):
        c = scores[baseline]
        reasons.append(f"{baseline}: R@5 {c.r5:.3f}, R@1 {c.r1:.3f}")
        if d.r5 < c.r5 - eps:
            reasons.append(
                f"{default} is below {baseline} on R@5: check the review and the {default} setup before changing models"
            )
            return Decision("investigate", tuple(reasons))

    qualified = []
    for name in challengers:
        if name not in scores:
            reasons.append(f"{name}: not run")
            continue
        s = scores[name]
        if len(s.hits5) != len(d.hits5):
            raise ValueError(f"{name} and {default} were scored on different queries")
        wins = sum(1 for x, y in zip(s.hits5, d.hits5) if x > y + eps)
        losses = sum(1 for x, y in zip(s.hits5, d.hits5) if x < y - eps)
        gain = s.r5 - d.r5
        failed = []
        if math.isnan(s.r5) or gain < rule.min_gain_r5 - eps:
            failed.append(f"R@5 gain {gain:+.3f} is below {rule.min_gain_r5:+.3f}")
        if math.isnan(s.r1) or s.r1 < d.r1 - eps:
            failed.append(f"R@1 {s.r1:.3f} is below {d.r1:.3f}")
        if wins <= losses:
            failed.append(f"wins {wins} queries and loses {losses} at hit@5")
        if not _within(s.image_ms, rule.max_image_ms):
            failed.append(f"image time {_fmt(s.image_ms, 'ms')} per crop is over {rule.max_image_ms:g} ms")
        if not _within(s.text_ms, rule.max_text_ms):
            failed.append(f"query time {_fmt(s.text_ms, 'ms')} is over {rule.max_text_ms:g} ms")
        if not _within(s.vram_mib, rule.max_vram_mib):
            failed.append(f"peak VRAM {_fmt(s.vram_mib, 'MiB')} is over {rule.max_vram_mib:g} MiB")
        summary = f"{name}: R@5 {s.r5:.3f} ({gain:+.3f}), R@1 {s.r1:.3f}, {wins} wins / {losses} losses"
        if failed:
            reasons.append(f"{summary}; does not qualify: " + "; ".join(failed))
        else:
            reasons.append(f"{summary}; qualifies")
            qualified.append((-s.r5, len(qualified), name))
    if qualified:
        winner = min(qualified)[2]
        reasons.append(f"{winner} has the highest R@5 of the challengers that qualify")
        return Decision(winner, tuple(reasons))
    reasons.append(f"no challenger qualifies, so {default} stays")
    return Decision(default, tuple(reasons))


def _fmt(value: float | None, unit: str) -> str:
    if value is None or math.isnan(value):
        return "(not measured)"
    return f"{value:.1f} {unit}"
