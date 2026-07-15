#!/usr/bin/env python3
from __future__ import annotations

import json

from driftguard.phase10.consistency_audit import run_consistency_audit


if __name__ == "__main__":
    print(json.dumps(run_consistency_audit(), indent=2, sort_keys=True))
