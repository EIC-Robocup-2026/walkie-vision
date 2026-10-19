"""Embed the data set with each candidate, time it, and score it.

Models run one at a time. Each one is loaded, used for every bucket, and
freed before the next is loaded, so the peak memory of one model does not
include another's.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

import metrics
from dataset import BUCKETS, Crop, Object, Query, similar_pairs

LATENCY_SAMPLES = 32
"""Single calls timed for each batch-1 latency. Fewer items are cycled through."""
WARMUP_CALLS = 3
LATENCY_BUCKET = 64
"""Batch-1 image latency is measured on crops of this size (or native, if none)."""


class Gpu:
    """CUDA memory and timing helpers; every method is a no-op off the GPU."""

    def __init__(self, device: str) -> None:
        self.torch = None
        if device.startswith("cuda"):
            import torch

            self.torch = torch
            self.device = torch.device(device)

    def reset(self) -> None:
        """Free cached memory and start a new peak."""
        if self.torch:
            self.torch.cuda.synchronize(self.device)
            # PyTorch keeps a cuBLAS workspace per handle and stream in its
            # allocator (8 MiB or more each, about 33 MiB here), and
            # empty_cache() does not free it. Clearing it makes each model's
            # peak include only its own workspace, and leftover_mib show
            # only real leaks. The function is private, so it is optional.
            clear = getattr(self.torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear is not None:
                clear()
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def sync(self) -> None:
        if self.torch:
            self.torch.cuda.synchronize(self.device)

    def allocated_mib(self) -> float | None:
        return self.torch.cuda.memory_allocated(self.device) / 2**20 if self.torch else None

    def peak_mib(self) -> float | None:
        return self.torch.cuda.max_memory_allocated(self.device) / 2**20 if self.torch else None


@dataclass
class Costs:
    load_s: float = math.nan
    weights_mib: float | None = None
    """Memory PyTorch had allocated once the model was loaded."""
    peak_mib: float | None = None
    """torch.cuda.max_memory_allocated over the whole run of this model."""
    image_ms: dict = field(default_factory=dict)
    """Bucket -> wall time per crop, embedding in batches."""
    image_ms_1: float | None = None
    """Median wall time to embed one crop on its own."""
    image_ms_1_p90: float | None = None
    """90th percentile of the same single calls."""
    text_ms: float = math.nan
    """Wall time per query, embedding in batches."""
    text_ms_1: float | None = None
    """Median wall time to embed one query on its own."""
    text_ms_1_p90: float | None = None
    """90th percentile of the same single calls."""
    leftover_mib: float | None = None
    """Memory still allocated after the model was freed; should be about 0."""


@dataclass
class ModelRun:
    name: str
    info: dict
    text_emb: np.ndarray
    bucket_emb: dict
    """Bucket -> one row per crop in that bucket."""
    costs: Costs


def _check_rows(name: str, emb: np.ndarray, n: int) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != n:
        raise RuntimeError(f"{name}: expected {n} embeddings, got shape {emb.shape}")
    if not np.isfinite(emb).all():
        raise RuntimeError(f"{name}: an embedding is not finite")
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"{name}: embeddings are not unit length (norms {norms.min():.4f} to {norms.max():.4f})")
    return emb


def embed_batched(fn: Callable, items: list, batch_size: int, gpu: Gpu, name: str) -> tuple[np.ndarray, float]:
    """Embed items in order, in batches; return the rows and the total wall time in ms."""
    rows, total = [], 0.0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        gpu.sync()
        t0 = time.perf_counter()
        out = fn(batch)
        gpu.sync()
        total += (time.perf_counter() - t0) * 1000
        rows.append(_check_rows(name, out, len(batch)))
    return np.concatenate(rows), total


def single_latency(fn: Callable, items: list, gpu: Gpu) -> tuple[float | None, float | None]:
    """Time LATENCY_SAMPLES single-item calls after a few warm-up calls.

    Returns the median and the 90th percentile in ms, or (None, None)
    without items. A small data set is cycled through, so that every model
    gets the same number of samples. The spread shows when a result near a
    limit of the decision rule is only timing noise.
    """
    if not items:
        return None, None
    for x in items[:WARMUP_CALLS]:
        fn([x])
    times = []
    for i in range(LATENCY_SAMPLES):
        x = items[i % len(items)]
        gpu.sync()
        t0 = time.perf_counter()
        fn([x])
        gpu.sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), statistics.quantiles(times, n=10, method="inclusive")[-1]


def run_candidate(cand, queries: list[Query], buckets: dict, device: str, batch_size: int, log=print) -> ModelRun:
    """Load one candidate, embed everything, measure it, and free it."""
    gpu = Gpu(device)
    costs = Costs()
    gpu.reset()
    t0 = time.perf_counter()
    cand.load(device)
    costs.load_s = time.perf_counter() - t0
    costs.weights_mib = gpu.allocated_mib()
    try:
        texts = [q.text for q in queries]
        first_images = next(ims for _, ims in buckets.values() if ims)
        # Warm-up: CUDA kernels, cuDNN and lazy initialisation.
        cand.embed_images(first_images[:batch_size])
        cand.embed_texts(texts[:batch_size])

        text_emb, ms = embed_batched(cand.embed_texts, texts, batch_size, gpu, cand.name)
        costs.text_ms = ms / len(texts)
        bucket_emb = {}
        for bucket, (_, images) in buckets.items():
            if not images:
                continue
            bucket_emb[bucket], ms = embed_batched(cand.embed_images, images, batch_size, gpu, cand.name)
            costs.image_ms[bucket] = ms / len(images)
            log(f"  {cand.name} {_bucket_name(bucket)}: {len(images)} crops, {costs.image_ms[bucket]:.1f} ms per crop")
        latency_images = buckets.get(LATENCY_BUCKET, ([], []))[1] or buckets["native"][1]
        costs.image_ms_1, costs.image_ms_1_p90 = single_latency(cand.embed_images, latency_images, gpu)
        costs.text_ms_1, costs.text_ms_1_p90 = single_latency(cand.embed_texts, texts, gpu)
        costs.peak_mib = gpu.peak_mib()
        info = cand.info()
        info["dim"] = int(text_emb.shape[1])
    finally:
        cand.close()
    gpu.reset()
    costs.leftover_mib = gpu.allocated_mib()
    return ModelRun(cand.name, info, text_emb, bucket_emb, costs)


def _bucket_name(bucket) -> str:
    return "native" if bucket == "native" else f"{bucket} px"


@dataclass
class Scores:
    summary: list[dict]
    per_query: list[dict]
    separation: list[dict]


def score_run(run: ModelRun, objects: list[Object], queries: list[Query], crops: list[Crop], buckets: dict) -> Scores:
    """Score one model's embeddings at every bucket and level."""
    summary, per_query, separation = [], [], []
    pairs = similar_pairs(objects)
    for bucket in BUCKETS:
        if bucket not in run.bucket_emb:
            continue
        idx = buckets[bucket][0]
        emb = run.bucket_emb[bucket]
        crop_objects = [crops[i].object_id for i in idx]
        present = set(crop_objects)
        scored = [(qi, q) for qi, q in enumerate(queries) if present.intersection(q.object_ids)]
        for level in metrics.LEVELS:
            scores, candidates = metrics.level_scores(level, run.text_emb, emb, crop_objects)
            ranked = []
            for qi, q in scored:
                r = metrics.rank_targets(scores[qi], candidates, set(q.object_ids))
                ranked.append(r)
                per_query.append({
                    "model": run.name,
                    "bucket": bucket,
                    "level": level,
                    "query": q.text,
                    "targets": " ".join(q.object_ids),
                    "rank": r.rank,
                    "tied": r.n_tied,
                    "hit1": r.hits[1],
                    "hit5": r.hits[5],
                    "top": r.top,
                    "top_score": r.top_score,
                    "best_target": r.best_target,
                    "best_target_score": r.best_target_score,
                })
            summary.append({
                "model": run.name,
                "bucket": bucket,
                "level": level,
                "n_queries": len(ranked),
                "n_objects": len(present),
                "n_crops": len(idx),
                "r1": metrics.recall_at_k(ranked, 1),
                "r5": metrics.recall_at_k(ranked, 5),
            })
        labels = np.asarray(crop_objects)
        for group, a, b in pairs:
            if a not in present or b not in present:
                continue
            s = metrics.pair_separation(emb[labels == a], emb[labels == b])
            separation.append({
                "model": run.name,
                "bucket": bucket,
                "pair": group,
                "a": a,
                "b": b,
                "n_a": s.n_a,
                "n_b": s.n_b,
                "within": s.within,
                "between": s.between,
                "margin": s.margin,
                "nn_accuracy": s.nn_accuracy,
            })
    return Scores(summary, per_query, separation)


def model_score(run: ModelRun, scores: Scores, rule: metrics.Rule) -> metrics.ModelScore:
    """Reduce one model's results to what the decision rule reads."""
    rows = [
        r for r in scores.summary
        if r["level"] == rule.level and r["bucket"] in rule.buckets and r["n_queries"] > 0
    ]
    hits = [
        r["hit5"] for r in scores.per_query if r["level"] == rule.level and r["bucket"] in rule.buckets
    ]
    image_ms = [run.costs.image_ms[b] for b in rule.buckets if b in run.costs.image_ms]
    return metrics.ModelScore(
        r1=float(np.mean([r["r1"] for r in rows])) if rows else math.nan,
        r5=float(np.mean([r["r5"] for r in rows])) if rows else math.nan,
        hits5=tuple(hits),
        image_ms=float(np.mean(image_ms)) if image_ms else None,
        text_ms=run.costs.text_ms_1,
        vram_mib=run.costs.peak_mib,
    )


def prepare_all(candidates: Sequence, allow_download: bool) -> None:
    """Check every model's files before any model is loaded."""
    from embedders import ModelUnavailable

    problems = []
    for cand in candidates:
        try:
            cand.prepare(allow_download)
        except ModelUnavailable as e:
            problems.append(f"{cand.name}: {e}")
    if problems:
        raise ModelUnavailable("\n  ".join(["missing model files:", *problems]))
