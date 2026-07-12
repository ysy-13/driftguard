from __future__ import annotations

from driftguard.sandbox.diff import path_is_allowed, state_diff


def test_empty_diff_and_dict_order():
    assert state_diff({"a": 1, "b": 2}, {"b": 2, "a": 1}) == []


def test_add_remove_replace_and_nested_stable_order():
    before = {"z": 1, "nested": {"a": 1, "remove": True}}
    after = {"z": 2, "nested": {"a": 1, "add": "x"}}
    assert state_diff(before, after) == [
        {"op": "remove", "path": "/nested/remove", "before": True},
        {"op": "add", "path": "/nested/add", "after": "x"},
        {"op": "replace", "path": "/z", "before": 1, "after": 2},
    ]


def test_clock_is_excluded_but_entity_timestamps_are_not():
    before = {"clock": "a", "item": {"updated_at": "a"}}
    after = {"clock": "b", "item": {"updated_at": "b"}}
    assert state_diff(before, after) == [
        {"op": "replace", "path": "/item/updated_at", "before": "a", "after": "b"}
    ]


def test_allowed_wildcard_matches_entity_root_and_children():
    assert path_is_allowed("/repositories/R1/issues/105", "/repositories/R1/issues/105/**")
    assert path_is_allowed("/repositories/R1/issues/105/title", "/repositories/R1/issues/105/**")
    assert not path_is_allowed("/repositories/R1/issues/106", "/repositories/R1/issues/105/**")
