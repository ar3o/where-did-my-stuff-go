import pytest

from whatchanged.evaluation import manual_scores as ms


def toy():
    key = {
        "events": {
            "a": dict(group=1, method="ours"),
            "b": dict(group=1, method="scenediff"),
            "c": dict(group=1, method="rules"),
            "d": dict(group=1, method="ours"),
            "e": dict(group=2, method="ours"),
            "f": dict(group=3, method="ours"),
        }
    }
    scores = {
        "a": dict(verdict="c", type_ok=True),
        "b": dict(verdict="d", dup_of="a"),  # same change found by another method
        "c": dict(verdict="c", type_ok=False),  # right object, wrong type
        "d": dict(verdict="w"),
        "e": dict(
            verdict="d", dup_of="a"
        ),  # not in the same group, ignored as a link to another method? same method here
        "f": dict(verdict="u"),
    }
    return scores, key


def test_wilson_interval_contains_the_estimate():
    p, lo, hi = ms.wilson(8, 10)
    assert lo < p < hi and p == pytest.approx(0.8)
    assert ms.wilson(0, 0)[0] != ms.wilson(0, 0)[0]


def test_parse_groups():
    assert ms.parse_groups("1-3,5") == {1, 2, 3, 5}


def test_duplicate_of_another_methods_correct_event_counts_as_found():
    scores, key = toy()
    r = ms.analyse(scores, key, {1})
    m = r["methods"]
    assert m["scenediff"]["duplicate_of_other_method"] == 1
    assert m["scenediff"]["precision"][0] == pytest.approx(1.0)
    assert m["ours"]["precision"][0] == pytest.approx(0.5)  # one correct, one wrong
    assert m["rules"]["wrong_type"] == 1
    assert r["pooled_changes"] == 2  # a (with b) and c
    assert m["ours"]["found"] == 1 and m["rules"]["found"] == 1


def test_unscored_groups_are_left_out():
    scores, key = toy()
    r = ms.analyse(scores, key, {1})
    assert r["n_events"] == 4
