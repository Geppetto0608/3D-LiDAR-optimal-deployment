# Expanded A/B Library Study

## Material Passport

- Mode: experiment implementation and functional verification.
- Date: 2026-11-23.
- Inputs: `configs/library_expanded_20261123.json`; common geometry/visibility.
- Status: implementation and dry-run available; formal performance results TBD.
- Existing experiments and library files are not modified or reused.

## Design

The previous query count was 7-20, not a constant ten. This extension builds a
new catalog, stores K=1..8, and varies both retrieved source count (1, 3) and
requested query budget (32, 64, 128).

- A: T and cross junctions, widths 8/10/12 m, center (50,50): six sources.
- B: 80 m finite corridors, widths 8/12 m, turns 0/30/60 degrees: six sources.
- Sources store up to 12 distinct feasible alternatives per K, including the
  best incumbent. These are multi-start greedy/swap alternatives, not exact
  top-12 solutions. Each alternative has its own gap against the source bound.
- A target grid: 5 m XY bins, height 3 m, relative yaw -30/0/30 degrees,
  pitch -10 degrees. Targets use the existing 0.5 m XY cells at height 1.5 m.
- B grid: 5 m station bins, both sides, the same height/yaw/pitch axes. Target
  station/lateral spacing is 2 m, height 1.5 m. A and B are different domains.
- Four target geometries, each open/restricted x clear/occluded: 16 cases.
- Source geometry does not duplicate target geometry. Targets remain synthetic
  development tests, not a held-out real-world or city-scale benchmark.
- Full eligible target domains have 174/210/192/288 poses before restrictions.
- Five method families: library, geometry single, geometry union, geometry
  portfolio MILP, and random (three seeds). Expected raw comparison: 5,376 rows.

The candidate grid and optimizer also change from the old K<=3 experiment;
do not attribute an old-to-new difference solely to source count. Inside this
study, source count and query budget are explicit factors with common physics.

## Transfer and Query Matching

Source ranking uses width/center distance for A and width/turn/module-length
distance for B, with fixed scales 4 m, 100 m and 60 degrees. It never uses target
coverage. A uses existing branch/side segment-fraction mapping. B uses existing
module-relative station mapping. Independent source layouts are not averaged.

The candidate order interleaves alternative rank, K, source, module, then the
stored candidate order. Duplicate poses are removed. Failed layout projections
are recorded; forbidden final locations are removed without repair. Surviving
locations remain usable even when their original full layout becomes illegal.

For a requested Q, actual q=min(Q, surviving unique library pool size). No random
or geometry padding is disguised as library output. All control methods query
the same actual q in that condition. The one-source pool is not required to be
a prefix of the three-source ranked pool; actual_q and pool_count are saved.

All query plans are frozen before physical raycasting. Every arm gets a fresh
visibility provider; the full reference runs only after all arm decisions.
Reference optimization never receives online layouts as warm starts. Source
masks and source optimality certificates are not transferred to target geometry.

## Solver and Certification

The existing packed coverage-cut MILP is reused, without changing its default
behavior for other callers. Expanded runs opt into bound-aware stopping:

| Role | MILP time cap per K |
|---|---:|
| Offline source | 1,800 s |
| Full eligible-domain reference | 1,800 s |
| Each queried-pool method | 300 s |
| Unoccluded geometry-portfolio planner | 10 s |

All use one solver thread, gap target zero, and preserve the best feasible
layout from smaller K. The optimization is at-most-K, not exactly K: the actual
sensor count is saved. If fewer sensors attain the domain coverage ceiling,
there is no forced padding with useless sensors.

With feasible count L and a valid upper bound U:

```text
relative_gap = (U - L) / L
near_optimal_certified = relative_gap <= 0.01
early_stop = runtime >= 300 s
             AND time_since_last_improvement >= 120 s
             AND relative_gap <= 0.01
```

An exact solver proof or a tight analytic coverage bound may finish earlier.
For a zero objective, the gap is zero only when the upper bound is also zero.
A flat coverage curve by itself never produces certification or early stopping.
At the time cap, a large gap remains uncertified. Numerical tolerance is 1e-5
target counts. Gurobi may exceed a time cap during a callback or shutdown.

`quality_time.csv/png` records best feasible coverage, upper bound and relative
gap for each K. Sampling occurs at solver callbacks, approximately every ten
seconds and at improvements/end; this is not a guaranteed periodic heartbeat
during presolve. The x-axis excludes mesh/raycast/warm-start/model preparation.
Analytically solved budgets have a time-zero point, not a convergence trace.

Queried-pool certificates apply only to that pool. Global finite-domain
`full_domain_near_optimal_certified` compares the method value with the full
reference upper bound. The reference may itself be time-limited:

```text
signed_incumbent_difference_pp = 100 * (L_reference - L_method) / N
true_loss_lower_pp = max(0, signed_incumbent_difference_pp)
true_loss_upper_pp = max(0, 100 * (U_reference - L_method) / N)
```

Thus a method exceeding the reference incumbent is not labeled a solver error
or a better-than-optimal deployment. No continuous-space certificate is claimed.

## Coverage, Overlap and Time

```text
d_j = number of selected sensors observing target j
coverage_pct = 100 * count(d_j >= 1) / N
overlap_pct = 100 * count(d_j >= 2) / N
overlap_of_covered_pct = 100 * count(d_j >= 2) / count(d_j >= 1)
excess_observations = sum(max(d_j - 1, 0))
```

The conditional overlap is null for zero coverage. Overlap is reported, not
penalized or constrained. It is neither automatically beneficial nor harmful.
The objective remains unweighted union coverage, not detection/tracking quality.

Reported times separate domain/mesh preparation, library load/transfer or
geometry planning, physical visibility, K-specific selection and solver-only
time. `online_through_k_sec` includes earlier K selection because the frontier
reuses previous feasible layouts. `online_all_k_sec` covers the whole frontier,
not a quantity to sum again over K rows. Source construction is separately
recorded in the catalog. Reference times are explicit comparison columns.

Shared preparation is charged once per hypothetical method; source mapping is
charged to each library query-budget evaluation even when computed once in the
worker. Independent validation, import/startup, artifact writes and plots are
outside online time. Source/case/worker wall times include additional work as
documented in JSON. This is sequential single-run timing with OS caches and
shared hardware, not controlled speedup evidence. Random seeds are averaged
within a condition before equally averaging conditions; A/B remain separate.

## Execution and Outputs

```bash
# Plan only: no mesh/raycast/optimization/files.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_expanded_study.py

# Tiny functional K1..8 run; deliberately short limits, NOT performance evidence.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_expanded_study.py \
  --smoke --output results/library_expanded_smoke_repeat --execute

# Formal, potentially long run. Must use a new output directory.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_expanded_study.py \
  --spec configs/library_expanded_20261123.json \
  --output results/library_expanded_20261123 --execute
```

The configured hard worker timeout is 48 hours. The sum of per-call caps is
2,023,680 seconds (not a runtime forecast), so completion within 48 hours is
not guaranteed. Exact/analytic early completion can reduce actual time greatly.
No automatic continuation or retry is implemented. Failures preserve completed
artifacts and report failed, not complete. Do not launch as a short smoke test.

```text
manifest.json / spec.json / source/ / worker.log
catalog.json / comparison.csv / summary.csv / comparison_<track>_s*_q*.png
library/<source>/entry.json, candidates.npz, frontier.json, quality_time.csv/png
cases/<target_condition>/query_plans.json, transfer_s*.json, candidates.npz,
    comparison.csv, validation.json
    <arm>/result.json, queried.npz, quality_time.csv/png
    reference/frontier.json, quality_time.csv/png
```

Source/target selected masks are independently recast using a fresh provider of
the same physical implementation. Stored density/overlap are checked against
the reference masks, and matched query counts are asserted. This is numerical
consistency validation, not an independent sensor simulator.

Tests: `tests/test_library_scalable.py`, `tests/test_library_expanded_study.py`,
existing `tests/test_coverage_cut_milp.py` and `tests/test_pipeline.py`.
Verification: 6+5+4+7 tests, syntax checks and three legacy CLI dry-runs passed.
Final functional execution `results/library_expanded_smoke_20261123_v3/` completed
two sources, eight target conditions, 1,152 raw rows and 240 seed-averaged summary
rows. All eight physical validations passed; frozen source hashes match the
tested current Python files. Earlier smoke directories are preserved. These
tiny short-limit runs do not establish performance or the expanded query counts.
No MAPPO, additional operational cost/spacing requirements, or detection models
are introduced. Generated results remain ignored by git.
