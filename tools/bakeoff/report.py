"""Write the results of a run: CSV files, report.md and run.json."""

from __future__ import annotations

import csv
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import metrics
from dataset import BUCKETS
from runner import LATENCY_SAMPLES

SUMMARY_COLUMNS = ("model", "bucket", "level", "n_queries", "n_objects", "n_crops", "r1", "r5")
PER_QUERY_COLUMNS = (
    "model", "bucket", "level", "query", "targets", "rank", "tied", "hit1", "hit5",
    "top", "top_score", "best_target", "best_target_score",
)
SEPARATION_COLUMNS = ("model", "bucket", "pair", "a", "b", "n_a", "n_b", "within", "between", "margin", "nn_accuracy")
COST_COLUMNS = (
    "model", "dtype", "dim", "load_s", "weights_mib", "peak_mib", "leftover_mib",
    "image_ms_1", "image_ms_1_p90", "text_ms_1", "text_ms_1_p90", "text_ms", *(f"image_ms_{b}" for b in BUCKETS),
)
PACKAGES = ("torch", "transformers", "sentence-transformers", "open-clip-torch", "ultralytics", "numpy", "pillow", "opencv-python")


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for row in rows:
            w.writerow([_cell(row.get(c)) for c in columns])


def cost_rows(runs) -> list[dict]:
    rows = []
    for run in runs:
        c = run.costs
        row = {
            "model": run.name,
            "dtype": run.info.get("dtype"),
            "dim": run.info.get("dim"),
            "load_s": c.load_s,
            "weights_mib": c.weights_mib,
            "peak_mib": c.peak_mib,
            "leftover_mib": c.leftover_mib,
            "image_ms_1": c.image_ms_1,
            "image_ms_1_p90": c.image_ms_1_p90,
            "text_ms_1": c.text_ms_1,
            "text_ms_1_p90": c.text_ms_1_p90,
            "text_ms": c.text_ms,
        }
        row.update({f"image_ms_{b}": c.image_ms.get(b) for b in BUCKETS})
        rows.append(row)
    return rows


def environment(device: str) -> dict:
    env = {
        "date_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "device": device,
        # Ultralytics sets this when it loads a model, and it changes cuBLAS
        # times and memory. The smoke run keeps YOLOE in its own process, so
        # it should be unset here unless the shell sets it.
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "packages": {},
    }
    for name in PACKAGES:
        try:
            env["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            env["packages"][name] = None
    if device.startswith("cuda"):
        import torch

        env["gpu"] = torch.cuda.get_device_name(torch.device(device))
        env["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(torch.device(device))
        env["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout
            env["driver"] = out.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            env["driver"] = None
    else:
        env["gpu"] = None
    return env


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def _spread(median, p90) -> str:
    """A batch-1 time as "median (p90)"."""
    if p90 is None:
        return fmt(median, 1)
    return f"{fmt(median, 1)} ({fmt(p90, 1)})"


def _bucket_label(bucket) -> str:
    return "native" if bucket == "native" else f"{bucket} px"


def _recall_table(summary: list[dict], models: list[str], level: str, key: str) -> list[str]:
    cell = {(r["model"], r["bucket"]): r for r in summary if r["level"] == level}
    buckets = [b for b in BUCKETS if any((m, b) in cell for m in models)]
    lines = [
        "| Model | " + " | ".join(_bucket_label(b) for b in buckets) + " |",
        "|---|" + "---:|" * len(buckets),
    ]
    for m in models:
        lines.append(f"| {m} | " + " | ".join(fmt(cell.get((m, b), {}).get(key)) for b in buckets) + " |")
    return lines


def markdown(
    env: dict,
    counts: dict,
    runs,
    summary: list[dict],
    per_query: list[dict],
    separation: list[dict],
    decision: metrics.Decision,
    rule: metrics.Rule,
    warnings: list[str],
    batch_size: int,
    smoke: bool,
) -> str:
    models = [r.name for r in runs]
    gpu = env.get("gpu") or "CPU"
    driver = f", driver {env['driver']}" if env.get("driver") else ""
    out = ["# Embedding bake-off", ""]
    if smoke:
        out += [
            "> **Smoke data.** This run used the two Ultralytics sample images. It shows that the",
            "> tool works; its numbers say nothing about which model is better.",
            "",
        ]
    out += [
        f"Run {env['date_utc']} on {env['host']}, {gpu}{driver}.",
        f"Data: {counts['objects']} objects, {counts['queries']} queries, {counts['crops']} crops "
        f"from {counts['images']} images.",
        "",
        "## Decision",
        "",
        f"**{decision.verdict}**",
        "",
        f"Rule: mean {rule.level} R@5 over {', '.join(f'{b} px' for b in rule.buckets)}. "
        f"A challenger needs R@5 at least {rule.min_gain_r5:+.2f} above {metrics.DEFAULT}, R@1 not below it, "
        f"more hit@5 wins than losses, at most {rule.max_image_ms:g} ms per crop in batches, "
        f"at most {rule.max_text_ms:g} ms per query, and at most {rule.max_vram_mib:g} MiB peak memory.",
        "",
        *[f"- {r}" for r in decision.reasons],
        "",
        "## Text to object (object_mean, the primary level)",
        "",
        "R@5:",
        "",
        *_recall_table(summary, models, "object_mean", "r5"),
        "",
        "R@1:",
        "",
        *_recall_table(summary, models, "object_mean", "r1"),
        "",
        "## Other levels",
        "",
        "object_max R@5 (an object scores its best crop):",
        "",
        *_recall_table(summary, models, "object_max", "r5"),
        "",
        "crop R@5 (every crop ranked on its own):",
        "",
        *_recall_table(summary, models, "crop", "r5"),
        "",
        "## Cost",
        "",
        f"Image times are wall time per crop, including preprocessing, in batches of {batch_size}. "
        f"Batch-1 times are the median of {LATENCY_SAMPLES} single calls, with the 90th percentile in brackets. "
        "A limit that lies between the two is within timing noise. Peak is torch.cuda.max_memory_allocated "
        "while the model ran. It includes the model's cuBLAS workspace and leaves out the CUDA context, "
        "which nvidia-smi counts.",
        "",
        "| Model | dtype | dim | load s | image ms/crop | image ms, batch 1 | query ms, batch 1 | peak MiB | weights MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        c = run.costs
        image_ms = [c.image_ms[b] for b in rule.buckets if b in c.image_ms]
        mean_ms = sum(image_ms) / len(image_ms) if image_ms else None
        out.append(
            f"| {run.name} | {run.info.get('dtype')} | {run.info.get('dim')} | {fmt(c.load_s, 1)} | {fmt(mean_ms, 1)} | "
            f"{_spread(c.image_ms_1, c.image_ms_1_p90)} | {_spread(c.text_ms_1, c.text_ms_1_p90)} | "
            f"{fmt(c.peak_mib, 0)} | {fmt(c.weights_mib, 0)} |"
        )
    out += ["", "## Similar objects", ""]
    if separation:
        out += [
            "Each cell is margin / nearest-neighbour accuracy. Margin is the mean cosine between views of one",
            "object minus the mean cosine between the two objects; compare it within one model only.",
            "",
        ]
        for bucket in ("native", 64):
            rows = [r for r in separation if r["bucket"] == bucket]
            if not rows:
                continue
            keys = list(dict.fromkeys((r["pair"], r["a"], r["b"]) for r in rows))
            cell = {(r["model"], r["pair"], r["a"], r["b"]): r for r in rows}
            out += [
                f"{_bucket_label(bucket)}:",
                "",
                "| Pair | " + " | ".join(models) + " |",
                "|---|" + "---:|" * len(models),
            ]
            for k in keys:
                vals = []
                for m in models:
                    r = cell.get((m, *k))
                    vals.append(f"{fmt(r['margin'])} / {fmt(r['nn_accuracy'], 2)}" if r else "-")
                out.append(f"| {k[0]}: {k[1]} vs {k[2]} | " + " | ".join(vals) + " |")
            out.append("")
    else:
        out += ["No pair has crops of both objects.", ""]

    out += [f"## Queries where the models disagree ({rule.level}, hit@5, {', '.join(f'{b} px' for b in rule.buckets)})", ""]
    hits = {}
    for r in per_query:
        if r["level"] == rule.level and r["bucket"] in rule.buckets:
            hits.setdefault((r["bucket"], r["query"]), {})[r["model"]] = r["hit5"]
    differ = [(k, v) for k, v in hits.items() if len({round(x, 6) for x in v.values()}) > 1]
    if differ:
        out += ["| Bucket | Query | " + " | ".join(models) + " |", "|---|---|" + "---:|" * len(models)]
        for (bucket, query), v in differ[:60]:
            out.append(f"| {bucket} px | {query} | " + " | ".join(fmt(v.get(m), 2) for m in models) + " |")
        if len(differ) > 60:
            out.append(f"\n{len(differ) - 60} more in per_query.csv.")
    else:
        out.append("None: every model hits or misses the same queries.")
    out += ["", "## Data", "", "| Bucket | Crops | Objects | Queries scored |", "|---|---:|---:|---:|"]
    first = models[0] if models else None
    for r in summary:
        if r["model"] == first and r["level"] == "object_mean":
            out.append(f"| {_bucket_label(r['bucket'])} | {r['n_crops']} | {r['n_objects']} | {r['n_queries']} |")
    if warnings:
        out += ["", "Warnings:", "", *[f"- {w}" for w in warnings]]
    out += [
        "",
        "Files: summary.csv, per_query.csv, separation.csv, costs.csv, run.json. "
        "See docs/guides/bakeoff.md for how to read them.",
        "",
    ]
    return "\n".join(out)


def write_all(results_dir: Path, *, env, counts, runs, scores, decision, rule, warnings, batch_size, smoke, inputs) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = [r for s in scores for r in s.summary]
    per_query = [r for s in scores for r in s.per_query]
    separation = [r for s in scores for r in s.separation]
    write_csv(results_dir / "summary.csv", SUMMARY_COLUMNS, summary)
    write_csv(results_dir / "per_query.csv", PER_QUERY_COLUMNS, per_query)
    write_csv(results_dir / "separation.csv", SEPARATION_COLUMNS, separation)
    write_csv(results_dir / "costs.csv", COST_COLUMNS, cost_rows(runs))
    (results_dir / "report.md").write_text(
        markdown(env, counts, runs, summary, per_query, separation, decision, rule, warnings, batch_size, smoke),
        encoding="utf-8",
    )
    record = {
        **env,
        "smoke": smoke,
        "batch_size": batch_size,
        "buckets": list(BUCKETS),
        "tie_tolerance": metrics.TIE_TOL,
        "rule": dataclasses.asdict(rule),
        "decision": {"verdict": decision.verdict, "reasons": list(decision.reasons)},
        "counts": counts,
        "inputs": {name: sha256_of(path) for name, path in inputs.items()},
        "models": {run.name: run.info for run in runs},
        "warnings": warnings,
    }
    (results_dir / "run.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
