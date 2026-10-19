import pytest
from scripts.replay_equivalence_shared_coverage import unit_credit, fractional_coverage
from validator.agent_v1.comparison_models import SilverUnit
from validator.agent_v1.silver_scoring import _coverage


def unit(ids, groups=None, importance="supporting", uid="u"):
    return SilverUnit(
        silver_unit_id=uid,
        paper_id="p",
        statement="Finding.",
        importance=importance,
        equivalent_candidate_ids=ids,
        metadata={"comparison_equivalence_group_ids": groups or []},
    )


def test_fraction_uses_only_retained_distinct_miners_not_reference_or_duplicates():
    u = unit(["b", "a1", "a2", "c"], ["g"])
    credits, audit = unit_credit(
        u,
        {"g": ["b", "a1", "a2", "c", "excluded"]},
        {"b": None, "a1": "A", "a2": "A", "c": "C", "excluded": "E"},
    )
    assert credits == {"A": 0.5, "C": 0.5}
    assert audit["n"] == 2
    assert sum(credits.values()) == 1


def test_merged_groups_have_one_pool_per_final_unit():
    u = unit(["a", "b", "a2", "c"], ["g1", "g2"])
    credits, audit = unit_credit(
        u,
        {"g1": ["a", "b"], "g2": ["a2", "c"]},
        {"a": "A", "a2": "A", "b": "B", "c": "C"},
    )
    assert audit["n"] == 3
    assert all(v == pytest.approx(1 / 3) for v in credits.values())
    assert sum(credits.values()) == pytest.approx(1)


def test_nongroup_contribution_keeps_full_credit_even_for_same_miner():
    credits, audit = unit_credit(
        unit(["a", "b", "a_independent", "c"], ["g"]),
        {"g": ["a", "b"]},
        {"a": "A", "b": "B", "a_independent": "A", "c": "C"},
        mixed_policy="max-credit",
    )
    assert credits == {"A": 1, "B": 0.5, "C": 1}
    assert audit["full_credit_overrides"] == ["A"]


def test_units_without_groups_keep_ordinary_coverage():
    u = unit(["a", "b"])
    credits, _ = unit_credit(u, {}, {"a": "A", "b": "B"})
    assert credits == {"A": 1, "B": 1}
    assert fractional_coverage([u], {"u": credits}, "A") == _coverage([u], ["u"])


def test_only_reference_members_have_no_miner_divisor():
    credits, audit = unit_credit(
        unit(["b"], ["g"]), {"g": ["b", "excluded"]}, {"b": None}
    )
    assert credits == {}
    assert audit["n"] == 0


def test_bad_provenance_is_rejected():
    with pytest.raises(ValueError, match="Unknown approved"):
        unit_credit(unit(["a"], ["unknown"]), {}, {"a": "A"})
    with pytest.raises(ValueError, match="owner is unknown"):
        unit_credit(unit(["missing"]), {}, {})


def test_fraction_keeps_importance_denominator_and_minor_cap():
    units = [
        unit([], importance="central", uid="c"),
        unit([], importance="minor", uid="m"),
    ]
    assert fractional_coverage(units, {"c": {"A": 0.5}, "m": {"A": 1}}, "A") == 0.525
    assert fractional_coverage(units, {"c": {"A": 1}, "m": {"A": 1}}, "A") == _coverage(
        units, ["c", "m"]
    )
    assert fractional_coverage(units, {"c": {}, "m": {}}, "A") == 0


def test_group_priority_prevents_second_claim_escaping_shared_credit():
    credits, audit = unit_credit(
        unit(["a", "b", "a_independent", "c"], ["g"]),
        {"g": ["a", "b"]},
        {"a": "A", "b": "B", "a_independent": "A", "c": "C"},
    )
    assert credits == {"A": 0.5, "B": 0.5, "C": 1}
    assert audit["full_credit_overrides"] == []


def test_missing_group_provenance_cannot_silently_avoid_fractional_credit():
    with pytest.raises(ValueError, match="does not match unit provenance"):
        unit_credit(unit(["a"]), {"g": ["a", "b"]}, {"a": "A"})
