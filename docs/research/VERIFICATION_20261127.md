# Publication Verification: 2026-11-27

This records checks performed before publishing the accumulated research source,
configs, protocols and compact audited result tables. It does not certify every
historical algorithm or rerun large performance experiments.

## Regression Command

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=tests .venv/bin/python -B -m unittest -q \
  test_research_summary_export test_library_mechanism \
  test_library_budget_study test_library_budget_report \
  test_rsu_stress test_rsu_stress_validation test_pipeline \
  test_optical_visibility test_library_scalable test_library_installation \
  test_library_expanded_study test_library_query_controls \
  test_library_control_evaluation test_library_transfer_evaluation \
  test_library_siting_report test_library_contract_audit \
  test_common_placement test_local_mappo test_prototype_offline \
  test_prototype_compatible test_shape_pilot test_installation_domain
```

Result: **167 tests, 166 passed, one skipped**, 66.07 seconds locally.
The skipped `test_common_placement.py` check requires explicit `--roadside-parity`
and full candidate construction; this optional expensive check was not enabled.
The executed tests include small real raycast/MILP/PPO fixtures, not full training.

## Other Checks

- All 148 pending Python files parsed successfully; pending JSON files decoded.
- Legacy GREEDY/MILP/MAPPO dry-runs passed on scenario 0, K=1 (MAPPO: 10 steps,
  one trial, dry-run only). No real training was launched by those commands.
- The nine published CSVs matched original audit SHA-256 hashes, byte lengths
  and row counts. The validation JSON excerpt matched the selected source fields.
- The research index's local links exist, and Markdown fences are balanced.
- Pending source files were screened for common private-key, GitHub-token,
  cloud-access-key, API-key and credential-URL patterns. No matches were found.
  This is a publication screen, not an exhaustive security audit.
- Raw `results/`, caches, logs, models, `.env` secrets and Gurobi license files
  are excluded from publication. Existing repository license terms are unchanged.

```bash
.venv/bin/python scripts/export_research_summary.py --check
git diff --check
```

The latest completed performance study remains the low-K cost study. The newer
mechanism study has passed its two-case/88-row smoke, but its formal 2,760-row
experiment and reserved-map evaluation have not run. GitHub publication does
not change those evidence boundaries.
