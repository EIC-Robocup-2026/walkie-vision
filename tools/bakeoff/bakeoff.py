"""Embedding bake-off: which model should make walkie-database's object and room vectors?

Subcommands, in the order they are used:

    extract   save frames from a rosbag2 recording into <workdir>/images
    propose   run YOLOE on the frames and write manifest.csv for review
    check     validate objects.csv, queries.csv and the reviewed manifest
    run       embed every crop with every candidate, then score and report

`run --smoke` builds a tiny data set from the Ultralytics sample images and
runs on it. docs/guides/bakeoff.md is the full guide.

Usage: pixi run bakeoff <subcommand> [options]  (run with --help for each)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bag import DEFAULT_TOPIC, BagError  # noqa: E402
from dataset import DatasetError, Workdir, load_dataset  # noqa: E402
from embedders import CANDIDATE_NAMES, ModelUnavailable  # noqa: E402

REPO = Path(os.environ.get("PIXI_PROJECT_ROOT") or HERE.parents[1])


def user_path(value: str) -> Path:
    """Resolve a path the way the user typed it: relative to where pixi was started."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(os.environ.get("INIT_CWD") or os.getcwd()) / path


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def yolo_device(device: str) -> str:
    """Ultralytics names GPUs by index: cuda -> 0, cuda:1 -> 1."""
    if device.startswith("cuda"):
        return device.partition(":")[2] or "0"
    return device


def cmd_extract(args) -> int:
    from bag import extract

    work = Workdir(args.workdir)
    result = extract(args.bag, args.topic, args.every, work.images, work.frames, args.prefix)
    print(f"{len(result.written)} of {result.messages} frames on {args.topic} saved to {work.images}")
    return 0


def cmd_propose(args) -> int:
    from dataset import read_objects
    from propose import propose, read_labels

    work = Workdir(args.workdir)
    objects = read_objects(work.objects)
    labels = read_labels(args.labels) if args.labels else list(dict.fromkeys(o.label for o in objects))
    print(f"YOLOE prompts: {', '.join(labels)}")
    crops = propose(
        work.images, objects, labels, work.manifest, work.review,
        conf=args.conf, device=yolo_device(args.device or default_device()), force=args.force,
    )
    blank = sum(1 for c in crops if not c.object_id)
    print(f"{len(crops)} boxes written to {work.manifest} ({blank} need an object_id); overlays in {work.review}")
    print("Review the manifest next: docs/guides/bakeoff.md, step 5.")
    return 0


def cmd_check(args) -> int:
    from dataset import cut_crops, make_buckets

    work = Workdir(args.workdir)
    objects, queries, crops, warnings = load_dataset(work)
    buckets = make_buckets(cut_crops(work.images, crops))
    print(f"{len(objects)} objects, {len(queries)} queries, {len(crops)} kept crops")
    for bucket, (idx, _) in buckets.items():
        present = {crops[i].object_id for i in idx}
        n_q = sum(1 for q in queries if present.intersection(q.object_ids))
        name = "native" if bucket == "native" else f"{bucket} px"
        print(f"  {name:>7}: {len(idx)} crops, {len(present)} objects, {n_q} queries")
    for w in warnings:
        print(f"warning: {w}")
    return 0


def run_bakeoff(work: Workdir, candidates: list, device: str, batch_size: int, allow_download: bool, smoke: bool = False):
    """Embed, score and report; return the decision. The tests call this with stand-in models."""
    import metrics
    import report
    import runner
    from dataset import cut_crops, make_buckets

    if batch_size < 1:
        raise DatasetError("--batch-size must be at least 1")
    objects, queries, crops, warnings = load_dataset(work)
    for w in warnings:
        print(f"warning: {w}")
    buckets = make_buckets(cut_crops(work.images, crops))
    counts = {
        "objects": len(objects),
        "queries": len(queries),
        "crops": len(crops),
        "images": len({c.image for c in crops}),
        "per_bucket": {str(b): len(idx) for b, (idx, _) in buckets.items()},
    }
    print(f"{counts['objects']} objects, {counts['queries']} queries, {counts['crops']} crops; device {device}")

    runner.prepare_all(candidates, allow_download)
    runs, scores = [], []
    for cand in candidates:
        print(f"{cand.name}: loading")
        run = runner.run_candidate(cand, queries, buckets, device, batch_size)
        runs.append(run)
        scores.append(runner.score_run(run, objects, queries, crops, buckets))
        c = run.costs
        print(
            f"{cand.name}: {run.info['dtype']}, dim {run.info['dim']}, load {c.load_s:.1f} s, "
            f"image {report.fmt(c.image_ms_1, 1)} ms and query {report.fmt(c.text_ms_1, 1)} ms at batch 1, "
            f"peak {report.fmt(c.peak_mib, 0)} MiB"
        )

    rule = metrics.Rule()
    decision = metrics.decide({r.name: runner.model_score(r, s, rule) for r, s in zip(runs, scores)}, rule)
    report.write_all(
        work.results,
        env=report.environment(device),
        counts=counts,
        runs=runs,
        scores=scores,
        decision=decision,
        rule=rule,
        warnings=warnings,
        batch_size=batch_size,
        smoke=smoke,
        inputs={"objects.csv": work.objects, "queries.csv": work.queries, "manifest.csv": work.manifest},
    )
    print(f"decision: {decision.verdict}")
    for reason in decision.reasons:
        print(f"  {reason}")
    print(f"report: {work.results / 'report.md'}")
    return decision


def cmd_run(args) -> int:
    if args.offline:
        # Must be set before huggingface_hub is imported.
        os.environ["HF_HUB_OFFLINE"] = "1"
    from embedders import all_candidates

    names = args.models.split(",") if args.models else list(CANDIDATE_NAMES)
    unknown = [n for n in names if n not in CANDIDATE_NAMES]
    if unknown:
        raise DatasetError(f"unknown model(s) {', '.join(unknown)}; choose from {', '.join(CANDIDATE_NAMES)}")
    device = args.device or default_device()
    if args.smoke:
        import smoke

        work = Workdir(args.workdir or smoke.default_workdir(REPO))
        smoke.build(work, yolo_device(device))
    elif args.workdir is None:
        raise DatasetError("pass --workdir, or --smoke")
    else:
        work = Workdir(args.workdir)
    candidates = [c for c in all_candidates() if c.name in names]
    run_bakeoff(work, candidates, device, args.batch_size, allow_download=not args.offline, smoke=args.smoke)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bakeoff", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="save frames from a rosbag2 recording")
    e.add_argument("bag", type=user_path, help="recording directory, or one .mcap or .db3 file")
    e.add_argument("--workdir", type=user_path, required=True)
    e.add_argument("--topic", default=DEFAULT_TOPIC, help=f"Image or CompressedImage topic (default {DEFAULT_TOPIC})")
    e.add_argument("--every", type=int, default=15, help="keep every N-th frame (default 15: one per second from the ZED at 15 Hz)")
    e.add_argument("--prefix", default=None, help="file name prefix (default: the recording's name)")
    e.set_defaults(func=cmd_extract)

    pr = sub.add_parser("propose", help="run YOLOE and write manifest.csv for review")
    pr.add_argument("--workdir", type=user_path, required=True)
    pr.add_argument("--labels", type=user_path, default=None, help="one YOLOE prompt per line (default: labels in objects.csv)")
    pr.add_argument("--conf", type=float, default=0.25, help="minimum detection score (default 0.25)")
    pr.add_argument("--device", default=None, help="cuda, cuda:N or cpu (default: cuda if there is a GPU)")
    pr.add_argument("--force", action="store_true", help="replace an existing manifest.csv")
    pr.set_defaults(func=cmd_propose)

    c = sub.add_parser("check", help="validate the data set without loading any model")
    c.add_argument("--workdir", type=user_path, required=True)
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="embed, score and write results/")
    r.add_argument("--workdir", type=user_path, default=None)
    r.add_argument("--smoke", action="store_true", help="build and use the sample data set (default workdir bakeoff-data/smoke)")
    r.add_argument("--models", default=None, help="comma-separated subset of the candidates (default: all four)")
    r.add_argument("--device", default=None, help="cuda, cuda:N or cpu (default: cuda if there is a GPU)")
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--offline", action="store_true", help="never download; fail if a model is not cached")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (DatasetError, BagError, ModelUnavailable, FileNotFoundError, FileExistsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
