# Research Snapshot: 2026-11-27

This is the GitHub-readable index for the accumulated LiDAR placement research.
The root [README](../../README.md) retains the detailed chronological record.
The repository's existing [LICENSE](../../LICENSE) remains unchanged; publishing
this snapshot does not grant an open-source license.

## Read First

| Material | Purpose |
|---|---|
| [Research status](../../RESEARCH_STATUS_20261127.md) | Environment, equations, A/B methods, results and TR-C evidence gaps |
| [Low-K protocol](../../LIBRARY_LOWK_COST_PROTOCOL.md) | Completed fixed-library cost/critical-coverage experiment |
| [Source/occlusion protocol](../../LIBRARY_RSU_STRESS_PROTOCOL.md) | 24-source library, static blockers, overlap constraints |
| [New mechanism protocol](../../LIBRARY_MECHANISM_PROTOCOL.md) | Implemented fixed-Q source/layout ablation; formal execution pending |
| [Research plan](../../TITS_RESEARCH_PLAN.md) | Historical T-ITS filename, current TR-C direction and unresolved work |
| [Storage contract](../../LIBRARY_STORAGE_PROTOCOL.md) | K-dependent feasible layouts, bounds and alternatives; historical first contract |

## Current Status

- A stores whole-topology layouts; B reuses corridor/component layouts. Both
  transfer candidates to a new geometry and perform joint constrained selection.
- The 24-source occlusion study and 16-case low-K cost study completed. Source
  entries are feasible solutions with explicit certificates, not all exact optima.
- Latest low-K audit: 1,288 online rows, 1,274 feasible and 14 infeasible; 184
  reference rows, 182 feasible and two infeasible. The two infeasible conditions
  have independent finite-domain critical-coverage upper-bound checks.
- Fresh target raycasting checked 40,840 masks; stored source archives and
  transfer/shortlist/constraint calculations were also checked. This validates
  consistency with the implemented model, not physical LiDAR detection accuracy.
- The new fixed-Q hybrid ablation has code/config/tests and a completed 88-row
  smoke. Its formal 2,760-row experiment has **not run**. Eight reserved map
  descriptors are excluded from its development runner.
- There is no claim of real-road generalization, continuous/global optimality,
  controlled speedup, certified minimum city-wide sensor count or RL necessity.

## Versioned Results

The small CSVs below are byte-identical copies from the completed audit. Their
hashes, row counts and source audit hash are in
[provenance.json](lowk_cost_20261124/provenance.json). The selected
[validation excerpt](lowk_cost_20261124/validation_excerpt.json) omits machine
paths and the large raw-input inventory. No experiment or audit is rerun by export.

| Table | Interpretation |
|---|---|
| [Method means](lowk_cost_20261124/method_summary.csv) | 182 common-feasible conditions; random seeds averaged within each condition |
| [Timing](lowk_cost_20261124/timing_summary.csv) | All 64 whole-frontier conditions, each counted once; not standalone-K latency |
| [Paired differences](lowk_cost_20261124/paired_summary.csv) | Library-diverse minus comparator, in coverage percentage points |
| [Map-balanced means](lowk_cost_20261124/map_balanced_summary.csv) | Equal geometry weighting, distinct from equal condition weighting |
| [Per-geometry differences](lowk_cost_20261124/paired_by_geometry.csv) | Geometry-level comparison, not independent K/seed samples |
| [Feasibility](lowk_cost_20261124/feasibility.csv) | Infeasible rows retained rather than converted to zero coverage |
| [Reference certificates](lowk_cost_20261124/references.csv) | All 184 bounds/gaps/statuses; gaps are ratios, not percentage points |
| [Infeasibility checks](lowk_cost_20261124/infeasibility_proofs.csv) | Use global_critical_upper_count, not the conditional one-group bound alone |
| [Case validation](lowk_cost_20261124/case_validation.csv) | 16 cases, direct ray counts and mask checksums |

Library-diverse mean coverage is 78.954%, versus geometry-portfolio 75.671%
(+3.283 percentage points). This is descriptive evidence on reused development
geometries, not uniform superiority. Of 182 feasible full-domain references,
109 have exact objective certificates and 143 have gaps within 1% (including
the exact cases); 39 remain above 1%. An online solution exceeding a reference
incumbent does not exceed the optimum. Timing is shared-server, single-run data.

The raw source layouts, candidate masks, checkpoints, logs and 3D HTML remain
under ignored `results/`. A clone can read these published summaries but cannot
independently reraycast them without regenerating/obtaining the original arrays.
Links into `results/` elsewhere in the chronological README are local-artifact
links, not downloads hosted by this repository.

## Reproduction And Verification

Install [requirements.txt](../../requirements.txt) in a virtual environment.
Gurobi experiments require a separate valid license; no license keys are included.
For a fresh source library, follow the source/occlusion protocol, then the low-K
protocol. New output roots are required. Current source and historical frozen
experiment snapshots are different provenance objects.

```bash
# No solve, training or physical query; requires the completed local source library.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_mechanism_study.py \
  --spec configs/library_mechanism_20261127.json

# Lightweight tests, including toy solves/raycast where dependencies are available.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_mechanism.py
.venv/bin/python tests/test_research_summary_export.py

# Read-only verification of published copies against the original local audit.
.venv/bin/python scripts/export_research_summary.py --check

# Re-export elsewhere; existing directories are refused. No optimization/raycast.
.venv/bin/python scripts/export_research_summary.py \
  --output results/research_summary_export_repeat --execute
```

The completed mechanism implementation was checked with 135 regression tests,
syntax checks, three legacy CLI dry-runs and the two-case smoke. Publication
adds an excerpt exporter and focused tests; verification does not mean every
historical large experiment has been rerun on the current checkout.
The [publication verification record](VERIFICATION_20261127.md) reports the wider
167-test check (166 passed, one explicitly opt-in parity test skipped) and the
exact command, summary-copy checks and publication scope.
