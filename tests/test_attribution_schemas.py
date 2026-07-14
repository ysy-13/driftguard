import json

from jsonschema import Draft202012Validator

from driftguard.diagnosis import DiagnosisEngine
from phase7_helpers import build_view


def test_evidence_trace_schema():
    schema = json.load(open("benchmark/schemas/evidence_trace_schema_v1.json"))
    Draft202012Validator(schema).validate(build_view("PD").trace.to_dict())


def test_diagnosis_result_schema():
    schema = json.load(open("benchmark/schemas/diagnosis_result_schema_v1.json"))
    result = DiagnosisEngine(build_view("PD")).diagnose().to_dict()
    Draft202012Validator(schema).validate(result)
