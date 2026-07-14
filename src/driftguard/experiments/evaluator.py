from __future__ import annotations


def strip_evaluator_fields(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}

