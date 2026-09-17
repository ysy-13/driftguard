# SpecDriftBench Artifact

This repository is the research artifact for **SpecDriftBench: A Matched
Benchmark for Distinguishing Tool-Specification Drift from Agent Errors and
Transient Failures in Tool-Using LLM Agents**.

The artifact contains the benchmark instances, prompts, JSON schemas, frozen
experiment configurations, scoring and analysis code, and the de-identified
records used for the paper. No API credential is required for validation,
scoring, replay of deterministic components, or inspection of the released
results.

## Artifact map

| Paper artifact | Repository location |
|---|---|
| Matched benchmark and ground truth | [`benchmark/matched_failures/`](benchmark/matched_failures/) |
| Drift families and task coverage | [`benchmark/drifts/`](benchmark/drifts/) |
| Agent-visible evidence views | [`benchmark/evidence_views/`](benchmark/evidence_views/) |
| Tool specification | [`benchmark/openapi/`](benchmark/openapi/) |
| Prompts | [`benchmark/prompts/`](benchmark/prompts/) |
| Output and record schemas | [`benchmark/schemas/`](benchmark/schemas/) |
| Frozen analysis plan | [`benchmark/analysis/`](benchmark/analysis/) |
| Frozen experiment configurations | [`configs/experiments/`](configs/experiments/) |
| Runtime and benchmark implementation | [`src/driftguard/`](src/driftguard/) |
| Scoring and analysis programs | [`scripts/`](scripts/) |
| Automated checks | [`tests/`](tests/) |
| Released experiment records and summaries | [`results/`](results/) |

The paper's formal model results come from the frozen V2 attempt:

```text
results/experiments/phase11/specdriftbench_component_heldout432/attempts/
  specdriftbench-heldout432-v2-20260719-3f4fd32-01/
```

Its `results/manifest.json` records the source commit, benchmark and prompt
fingerprints, provider configurations, budgets, and execution metadata. Its
`results/records/` directory contains 432 de-identified records, and
`results/summary.json` contains the run-level summary. Development runs and the
incomplete V1 attempt are not used as formal model-performance evidence.

## Environment

The package supports Python 3.9 or newer. From a fresh clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The released artifact does not require the local `.env` file. Real-provider
execution is deliberately fail-closed and is not part of the offline
reproduction path.

## Validate the benchmark and implementation

Run the benchmark validators and complete automated test suite:

```bash
python scripts/validate_openapi.py
python scripts/validate_tasks.py
python scripts/validate_drifts.py
python scripts/validate_matched_failures.py
python scripts/run_oracle.py
pytest -q
```

Validate the frozen held-out protocol without making provider calls:

```bash
python scripts/run_specdriftbench_component_heldout.py --validate-config
```

The deterministic conformance reports can be regenerated offline with:

```bash
python scripts/run_injection_conformance.py
python scripts/run_attribution_conformance.py
python scripts/run_healing_conformance.py
```

## Reproduce the paper analysis

The primary statistical analysis is deterministic, performs no provider calls,
and validates the frozen source hashes before computing any result:

```bash
python scripts/analyze_specdriftbench_experiment2.py --analysis-only
```

The analysis produces the evidence-view metrics, paired comparisons,
family-clustered bootstrap intervals, baseline results, and paper-ready tables
under the formal attempt's `analysis/experiment2/` directory. Figures can be
regenerated after installing the optional Pillow and ReportLab dependencies:

```bash
python -m pip install Pillow reportlab
python scripts/analyze_specdriftbench_experiment2.py --render-only
```

## Released and excluded material

The release includes normalized model predictions, usage metadata, manifests,
summaries, and derived analyses needed to audit the reported results. It does
not include API keys, local `.env` files, provider-side account data, hidden
chain-of-thought, or transient cache/checkpoint files. The code never requires
chain-of-thought for scoring.

Provider calls are unnecessary to check the released claims. Re-running the
original paid calls would require users to supply their own credentials and
explicitly pass the real-API authorization gates; it may also differ as hosted
models change over time.

## Citation and archival version

The GitHub repository is the development home of the artifact. For the
camera-ready paper, cite a tagged release (and an archival DOI, if one is
created) rather than the moving `main` branch. The release tag should be added
here and to the paper only after the final artifact commit has been pushed.

## Contact

Artifact questions may be opened as issues in this repository. Author and
institutional contact information will be included in the camera-ready paper.
