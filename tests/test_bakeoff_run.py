"""The whole `run` pipeline of the embedding bake-off, on the CPU, with stand-in models."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

_TOOL = str(Path(__file__).resolve().parents[1] / "tools" / "bakeoff")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

import bakeoff  # noqa: E402
import runner  # noqa: E402
from dataset import Crop, Object, Query, Workdir, write_manifest, write_objects, write_queries  # noqa: E402
from embedders import Candidate, ModelUnavailable  # noqa: E402

COLOURS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}
OBJECTS = [
    Object("red_a", "box", "twins"),
    Object("red_b", "box", "twins"),
    *(Object(c, "box") for c in ("green", "blue", "yellow", "cyan", "magenta")),
]
QUERIES = [Query("red", ("red_a", "red_b")), *(Query(c, (c,)) for c in ("green", "blue", "yellow", "cyan", "magenta"))]
# Object -> (colour, box in frame 1, box in frame 2). Longest sides:
# red_a 100, red_b 40, green 70, blue 120, yellow 50, cyan 60, magenta 36.
LAYOUT = {
    "red_a": ("red", (10, 10, 110, 70), (200, 200, 300, 260)),
    "red_b": ("red", (130, 10, 170, 40), (20, 250, 60, 280)),
    "green": ("green", (190, 10, 260, 60), (300, 100, 370, 150)),
    "blue": ("blue", (10, 90, 130, 190), (60, 140, 180, 240)),
    "yellow": ("yellow", (150, 90, 200, 110), (300, 20, 350, 40)),
    "cyan": ("cyan", (220, 90, 280, 130), (200, 20, 260, 60)),
    "magenta": ("magenta", (300, 10, 336, 46), (100, 20, 136, 56)),
}


def _colour_vector(rgb) -> np.ndarray:
    v = np.asarray(rgb, dtype=np.float32)
    return v / np.linalg.norm(v)


class Fake(Candidate):
    """Embeds an image as its mean colour, and a query as the colour it names.

    With inverted=True the image vectors point the other way, so the model
    ranks the right objects last.
    """

    live: set = set()
    log: list = []

    def __init__(self, name, inverted=False, available=True):
        super().__init__()
        self.name, self.inverted, self.available = name, inverted, available

    def prepare(self, allow_download):
        Fake.log.append(("prepare", self.name))
        if not self.available:
            raise ModelUnavailable("not in the cache")

    def load(self, device):
        assert not Fake.live, f"{self.name} loaded while {Fake.live} is still loaded"
        Fake.live.add(self.name)
        Fake.log.append(("load", self.name))

    def embed_images(self, images):
        rows = []
        for im in images:
            v = _colour_vector(np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(axis=0))
            rows.append(-v if self.inverted else v)
        return np.stack(rows)

    def embed_texts(self, texts):
        return np.stack([_colour_vector(COLOURS[t]) for t in texts])

    def info(self):
        return {"repo": "fake", "revision": "0", "licence": "-", "dtype": "float32"}

    def close(self):
        Fake.live.discard(self.name)
        Fake.log.append(("close", self.name))


@pytest.fixture
def work(tmp_path):
    work = Workdir(tmp_path / "work")
    work.images.mkdir(parents=True)
    crops = []
    for frame in (1, 2):
        im = Image.new("RGB", (400, 300), (128, 128, 128))
        draw = ImageDraw.Draw(im)
        for oid, (colour, *boxes) in LAYOUT.items():
            box = boxes[frame - 1]
            # PIL rectangles include the end pixel; crop boxes exclude it.
            draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=COLOURS[colour])
            crops.append(Crop(f"f{frame}_{oid}", f"f{frame}.png", box, oid, "box", 0.9))
        im.save(work.images / f"f{frame}.png")
    write_objects(work.objects, OBJECTS)
    write_queries(work.queries, QUERIES)
    write_manifest(work.manifest, crops)
    Fake.live.clear()
    Fake.log.clear()
    return work


def _candidates(clip_inverted=True):
    return [
        Fake("clip_vit_b16", inverted=clip_inverted),
        Fake("siglip2_b16_256", inverted=not clip_inverted),
        Fake("eg2_t70"),
        Fake("eg2_t140"),
    ]


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_run_writes_every_result(work):
    decision = bakeoff.run_bakeoff(work, _candidates(), "cpu", batch_size=3, allow_download=False)
    # EG2 is no better than SigLIP 2 here, and no memory is measured on the CPU.
    assert decision.verdict == "siglip2_b16_256"
    assert any(r.startswith("eg2_t70") and "(not measured)" in r for r in decision.reasons)

    # Every model was prepared before the first load, and each was closed
    # before the next one was loaded.
    names = [c.name for c in _candidates()]
    assert Fake.log == [("prepare", n) for n in names] + [(step, n) for n in names for step in ("load", "close")]

    summary = _rows(work.results / "summary.csv")
    assert len(summary) == 4 * 5 * 3  # models x buckets x levels
    cell = {(r["model"], r["bucket"], r["level"]): r for r in summary}
    for level in ("object_mean", "object_max", "crop"):
        assert float(cell[("siglip2_b16_256", "native", level)]["r1"]) == 1.0
    assert float(cell[("clip_vit_b16", "32", "object_mean")]["r5"]) == 0.0
    # A bucket holds only the crops whose longest side is at least its size.
    assert [cell[("eg2_t70", b, "crop")]["n_crops"] for b in ("native", "32", "48", "64", "96")] == [
        "14", "14", "10", "6", "4",
    ]
    assert cell[("eg2_t70", "96", "object_mean")]["n_objects"] == "2"
    assert cell[("eg2_t70", "96", "object_mean")]["n_queries"] == "2"

    per_query = _rows(work.results / "per_query.csv")
    red = [r for r in per_query if r["model"] == "eg2_t70" and r["bucket"] == "native" and r["query"] == "red"]
    # Two targets with the same score are not a tie with a non-target.
    assert {(r["level"], r["rank"], r["tied"], float(r["hit1"])) for r in red} == {
        ("object_mean", "1", "0", 1.0),
        ("object_max", "1", "0", 1.0),
        ("crop", "1", "0", 1.0),
    }

    sep = _rows(work.results / "separation.csv")
    twins = [r for r in sep if r["model"] == "eg2_t70" and r["bucket"] == "native"]
    assert [(r["pair"], r["a"], r["b"], r["n_a"], r["n_b"]) for r in twins] == [("twins", "red_a", "red_b", "2", "2")]
    # Same colour: each view is as close to the other object as to its own.
    assert float(twins[0]["margin"]) == pytest.approx(0.0, abs=1e-6)
    assert float(twins[0]["nn_accuracy"]) == 0.5
    # red_b is under 48 px, so the pair is not scored there.
    assert not [r for r in sep if r["bucket"] == "48"]

    costs = {r["model"]: r for r in _rows(work.results / "costs.csv")}
    assert set(costs) == set(names)
    assert costs["eg2_t70"]["peak_mib"] == ""
    assert float(costs["eg2_t70"]["image_ms_1"]) >= 0
    assert float(costs["eg2_t70"]["text_ms_1_p90"]) >= float(costs["eg2_t70"]["text_ms_1"])
    assert costs["eg2_t70"]["dim"] == "3"

    report = (work.results / "report.md").read_text()
    assert "**siglip2_b16_256**" in report
    assert "| twins: red_a vs red_b |" in report
    assert "Smoke data" not in report
    record = json.loads((work.results / "run.json").read_text())
    assert record["decision"]["verdict"] == "siglip2_b16_256"
    assert record["counts"]["crops"] == 14
    assert record["counts"]["per_bucket"] == {"native": 14, "32": 14, "48": 10, "64": 6, "96": 4}
    assert record["rule"]["min_gain_r5"] == 0.05
    assert set(record["inputs"]) == {"objects.csv", "queries.csv", "manifest.csv"}
    assert record["models"]["eg2_t70"]["dim"] == 3


def test_run_flags_siglip2_below_clip(work):
    decision = bakeoff.run_bakeoff(work, _candidates(clip_inverted=False), "cpu", batch_size=16, allow_download=False)
    assert decision.verdict == "investigate"


def test_run_with_some_models(work):
    decision = bakeoff.run_bakeoff(work, _candidates()[:2], "cpu", batch_size=16, allow_download=False)
    assert decision.verdict == "siglip2_b16_256"
    assert "eg2_t70: not run" in decision.reasons
    assert {r["model"] for r in _rows(work.results / "summary.csv")} == {"clip_vit_b16", "siglip2_b16_256"}


def test_missing_model_fails_before_any_load(work):
    cands = _candidates()
    cands[2] = Fake("eg2_t70", available=False)
    with pytest.raises(ModelUnavailable, match="eg2_t70: not in the cache"):
        bakeoff.run_bakeoff(work, cands, "cpu", batch_size=4, allow_download=False)
    assert [step for step, _ in Fake.log] == ["prepare"] * 4
    assert not work.results.exists()


def test_single_latency_takes_the_same_number_of_samples_from_few_items():
    calls = []
    median, p90 = runner.single_latency(calls.append, ["a", "b"], runner.Gpu("cpu"))
    # Two warm-up calls (one per item), then the items cycled through.
    assert calls == [["a"], ["b"]] + [["a"], ["b"]] * (runner.LATENCY_SAMPLES // 2)
    assert 0 <= median <= p90
    assert runner.single_latency(calls.append, [], runner.Gpu("cpu")) == (None, None)


def test_bad_embeddings_are_refused():
    class Broken(Fake):
        def embed_images(self, images):
            return np.zeros((len(images), 3), np.float32)

    with pytest.raises(RuntimeError, match="not unit length"):
        runner.embed_batched(Broken("x").embed_images, [Image.new("RGB", (4, 4))], 2, runner.Gpu("cpu"), "x")
    with pytest.raises(RuntimeError, match="expected 2 embeddings"):
        runner.embed_batched(lambda b: np.eye(3)[:1], [1, 2], 2, runner.Gpu("cpu"), "x")


def test_cli_check_and_errors(work, capsys):
    assert bakeoff.main(["check", "--workdir", str(work.root)]) == 0
    out = capsys.readouterr().out
    assert "7 objects, 6 queries, 14 kept crops" in out
    assert "  96 px: 4 crops, 2 objects, 2 queries" in out

    # Blank the object_id of the first data row.
    rows = work.manifest.read_text().splitlines()
    rows[1] = rows[1].rsplit(",", 1)[0] + ","
    work.manifest.write_text("\n".join(rows) + "\n")
    assert bakeoff.main(["check", "--workdir", str(work.root)]) == 1
    assert "1 row(s) not reviewed" in capsys.readouterr().err

    assert bakeoff.main(["run", "--workdir", str(work.root), "--models", "clip,eg2"]) == 1
    assert "unknown model(s) clip, eg2" in capsys.readouterr().err
