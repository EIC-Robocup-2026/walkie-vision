"""Metrics of the embedding bake-off (tools/bakeoff/metrics.py), on hand-made vectors."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOL = str(Path(__file__).resolve().parents[1] / "tools" / "bakeoff")
if _TOOL not in sys.path:
    sys.path.insert(0, _TOOL)

import metrics  # noqa: E402
from metrics import ModelScore, Rule, decide, hit_probability, level_scores, pair_separation, rank_targets  # noqa: E402


def test_rank_without_ties():
    r = rank_targets([0.9, 0.1, 0.5, 0.3], ["a", "b", "c", "d"], {"c"}, ks=(1, 2, 5))
    assert r.rank == 2
    assert r.n_tied == 0
    assert r.hits == {1: 0.0, 2: 1.0, 5: 1.0}
    assert (r.top, r.top_score) == ("a", 0.9)
    assert (r.best_target, r.best_target_score) == ("c", 0.5)


def test_rank_with_ties_matches_hand_computation():
    # One target at 0.5; non-targets at 0.9, 0.5 and 0.5. One non-target is
    # above it, and the target is ordered at random among three tied
    # candidates: it lands 2nd, 3rd or 4th with chance 1/3 each.
    r = rank_targets([0.5, 0.9, 0.5, 0.5], ["t", "x", "y", "z"], {"t"}, ks=(1, 2, 3, 4))
    assert r.rank == 2
    assert r.n_tied == 2
    assert r.hits[1] == 0.0
    assert r.hits[2] == pytest.approx(1 / 3)
    assert r.hits[3] == pytest.approx(2 / 3)
    assert r.hits[4] == 1.0


def test_hit_probability_formula():
    assert hit_probability(n_above=1, n_tied_targets=1, n_tied_others=2, k=2) == pytest.approx(1 / 3)
    # Two tied targets and one tied non-target fill one place: 1 - 1/3.
    assert hit_probability(n_above=0, n_tied_targets=2, n_tied_others=1, k=1) == pytest.approx(2 / 3)
    # More places than tied non-targets: a target always gets in.
    assert hit_probability(n_above=0, n_tied_targets=1, n_tied_others=3, k=4) == 1.0
    assert hit_probability(n_above=5, n_tied_targets=1, n_tied_others=0, k=5) == 0.0
    with pytest.raises(ValueError):
        hit_probability(0, 0, 0, 1)


def test_ties_within_tolerance():
    # Float noise far below the tolerance still counts as a tie.
    r = rank_targets([0.5, 0.5 + 1e-9], ["t", "x"], {"t"})
    assert r.n_tied == 1
    assert r.hits[1] == pytest.approx(0.5)


def test_multi_target_query_uses_the_best_target():
    r = rank_targets([0.9, 0.2, 0.5, 0.4], ["a", "b", "c", "d"], {"b", "d"}, ks=(1, 2, 3))
    assert r.best_target == "d"
    assert r.rank == 3
    assert r.hits == {1: 0.0, 2: 0.0, 3: 1.0}


def test_two_tied_targets_and_one_tied_other():
    r = rank_targets([0.7, 0.7, 0.7, 0.1], ["t1", "t2", "x", "y"], {"t1", "t2"})
    assert r.hits[1] == pytest.approx(2 / 3)
    assert r.hits[5] == 1.0


def test_rank_needs_a_target():
    with pytest.raises(ValueError):
        rank_targets([0.1, 0.2], ["a", "b"], {"c"})


def test_recall_at_k_averages_queries():
    ranked = [
        rank_targets([0.9, 0.1], ["a", "b"], {"a"}),  # hit at 1
        rank_targets([0.9, 0.1], ["a", "b"], {"b"}),  # hit at 2 only
        rank_targets([0.5, 0.5], ["a", "b"], {"a"}),  # tie: 0.5 at 1
    ]
    assert metrics.recall_at_k(ranked, 1) == pytest.approx((1 + 0 + 0.5) / 3)
    assert metrics.recall_at_k(ranked, 5) == 1.0
    assert math.isnan(metrics.recall_at_k([], 1))


def test_levels_differ_on_a_split_object():
    # Object A has two very different views; object B has one view.
    crops = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    objects = ["A", "A", "B"]
    query = np.array([[1.0, 0.0]])

    s, cands = level_scores("object_mean", query, crops, objects)
    assert cands == ["A", "B"]
    np.testing.assert_allclose(s, [[math.sqrt(0.5), 0.8]])
    assert rank_targets(s[0], cands, {"A"}).hits[1] == 0.0  # the mean of A points away from the query

    s, cands = level_scores("object_max", query, crops, objects)
    np.testing.assert_allclose(s, [[1.0, 0.8]])
    assert rank_targets(s[0], cands, {"A"}).hits[1] == 1.0

    s, cands = level_scores("crop", query, crops, objects)
    assert cands == ["A", "A", "B"]
    np.testing.assert_allclose(s, [[1.0, 0.0, 0.8]])
    assert rank_targets(s[0], cands, {"A"}).hits[1] == 1.0


def test_crop_level_counts_every_crop_of_other_objects():
    # Five views of a non-target object push the only target crop out of
    # the top 5 crops, though only one object is above it.
    crops = np.array([[0.6, 0.8]] + [[1.0, 0.0]] * 5)
    objects = ["T"] + ["X"] * 5
    query = np.array([[1.0, 0.0]])
    s, cands = level_scores("crop", query, crops, objects)
    crop = rank_targets(s[0], cands, {"T"})
    assert crop.rank == 6
    assert crop.hits[5] == 0.0
    s, cands = level_scores("object_mean", query, crops, objects)
    assert rank_targets(s[0], cands, {"T"}).hits[5] == 1.0


def test_object_vector_is_the_renormalised_mean():
    v, objects = metrics.object_vectors(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), ["A", "A", "B"])
    assert objects == ["A", "B"]
    np.testing.assert_allclose(v, [[math.sqrt(0.5), math.sqrt(0.5)], [0.0, 1.0]])


def test_unknown_level_and_zero_vector():
    with pytest.raises(ValueError):
        level_scores("image", np.eye(2), np.eye(2), ["a", "b"])
    with pytest.raises(ValueError):
        metrics.normalize(np.zeros((1, 3)))


def test_pair_separation_by_hand():
    a = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    s = pair_separation(a, b)
    assert (s.n_a, s.n_b) == (2, 1)
    assert s.within == pytest.approx(0.8)  # the one pair of A's views
    assert s.between == pytest.approx(0.3)  # mean of 0 and 0.6
    assert s.margin == pytest.approx(0.5)
    # Each view of A is closer to A's other view (0.8) than to B (0 and 0.6);
    # B has a single view, so it does not vote.
    assert s.nn_accuracy == 1.0


def test_pair_separation_counts_a_tie_as_half():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.0, 1.0]])
    s = pair_separation(a, b)
    assert s.within == pytest.approx(0.0)
    assert s.between == pytest.approx(0.5)
    assert s.margin == pytest.approx(-0.5)
    # View 1: 0 to its sibling, 0 to B (tie, 0.5). View 2: 0 to its sibling, 1 to B (0).
    assert s.nn_accuracy == pytest.approx(0.25)


def test_pair_separation_with_single_views_is_undefined():
    s = pair_separation(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert math.isnan(s.within) and math.isnan(s.margin) and math.isnan(s.nn_accuracy)
    assert s.between == pytest.approx(0.0)


def _score(r5, r1=None, hits5=(1.0, 0.0), image_ms=5.0, text_ms=20.0, vram_mib=1500.0):
    return ModelScore(r1=r5 if r1 is None else r1, r5=r5, hits5=hits5, image_ms=image_ms, text_ms=text_ms, vram_mib=vram_mib)


def test_decide_keeps_siglip2_without_a_clear_gain():
    d = decide({
        "clip_vit_b16": _score(0.60),
        "siglip2_b16_256": _score(0.70, hits5=(1.0, 0.0)),
        "eg2_t70": _score(0.73, hits5=(1.0, 1.0)),  # +0.03 only
        "eg2_t140": _score(0.74, hits5=(1.0, 1.0)),
    })
    assert d.verdict == "siglip2_b16_256"
    assert any("eg2_t70" in r and "gain" in r for r in d.reasons)


def test_decide_picks_a_challenger_that_clears_every_bar():
    d = decide({
        "clip_vit_b16": _score(0.60),
        "siglip2_b16_256": _score(0.70, hits5=(0.0, 0.0, 1.0)),
        "eg2_t70": _score(0.80, hits5=(1.0, 0.0, 1.0)),
        "eg2_t140": _score(0.80, hits5=(0.0, 1.0, 1.0)),
    })
    assert d.verdict == "eg2_t70"  # the tie goes to the lower budget


def test_decide_prefers_the_higher_recall_challenger():
    d = decide({
        "siglip2_b16_256": _score(0.70, hits5=(0.0, 0.0)),
        "eg2_t70": _score(0.76, hits5=(1.0, 0.0)),
        "eg2_t140": _score(0.85, hits5=(1.0, 1.0)),
    })
    assert d.verdict == "eg2_t140"


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"image_ms": 25.0}, "image time"),
        ({"text_ms": 80.0}, "query time"),
        ({"vram_mib": 4000.0}, "peak VRAM"),
        ({"vram_mib": None}, "not measured"),
        ({"r1": 0.5}, "R@1 0.500 is below"),
        ({"hits5": (0.0, 1.0)}, "wins 0"),
    ],
)
def test_decide_rejects_a_challenger_that_fails_one_bar(change, reason):
    challenger = {"r5": 0.85, "r1": 0.85, "hits5": (1.0, 1.0), "image_ms": 5.0, "text_ms": 20.0, "vram_mib": 1500.0}
    challenger.update(change)
    d = decide({
        "siglip2_b16_256": _score(0.70, r1=0.70, hits5=(0.0, 1.0)),
        "eg2_t70": ModelScore(**challenger),
    })
    assert d.verdict == "siglip2_b16_256"
    assert any(r.startswith("eg2_t70") and reason in r for r in d.reasons)


def test_decide_investigates_when_siglip2_is_below_clip():
    d = decide({"clip_vit_b16": _score(0.80), "siglip2_b16_256": _score(0.70), "eg2_t70": _score(0.95)})
    assert d.verdict == "investigate"


def test_decide_needs_siglip2():
    assert decide({"clip_vit_b16": _score(0.8)}).verdict == "incomplete"
    assert decide({"siglip2_b16_256": _score(math.nan)}).verdict == "incomplete"


def test_decide_rejects_misaligned_queries():
    with pytest.raises(ValueError):
        decide({"siglip2_b16_256": _score(0.7, hits5=(1.0,)), "eg2_t70": _score(0.9, hits5=(1.0, 1.0))})


def test_rule_constants_are_the_documented_ones():
    # docs/guides/bakeoff.md quotes these values; change both together.
    assert Rule() == Rule(
        level="object_mean", buckets=(32, 48, 64), min_gain_r5=0.05, max_image_ms=10.0, max_text_ms=50.0, max_vram_mib=3072.0
    )
