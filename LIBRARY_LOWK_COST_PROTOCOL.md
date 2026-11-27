# Low-K Frozen-Library Cost and Critical-Coverage Study

## Material Passport

- Mode: code experiment implementation and execution.
- Date: 2026-11-24. Results: **TBD until the manifest is complete**.
- Inputs: validated `results/library_rsu_stress_20261123` source library only.
- No detector, RL, real installation prices, or operational RSU compliance claim.

## Research Question

Does a fixed geometric deployment library remain useful when sensor budgets are
scarce and a new map adds site-dependent cost and critical-region requirements?
Do not assume library or reinforcement-learning superiority. The eight target
geometries and obstacle seeds were used previously and are development maps.

## Fixed Design

- Preserve all 24 source maps, their K1..8 union solutions and K4/8 redundancy
  alternatives, including their original bounds and uncertified solutions.
  No source reoptimization or cost-aware relabeling is performed.
- A: T/CROSS, 120/180m square maps, K=2,3,4.
- B: 160/240/320m connected corridors, K=3,4,6; 480m stress case K=6,8.
- Static scene levels: mixed and dense. Each scene is optimized independently;
  this is NOT one robust deployment optimized over multiple scenes.
- Same target support and obstacles as the preceding study. A uses 0.5m road
  sampling, B uses 1m station sampling; target heights are 1/1.5/2m.
  Installation bins are 4m; heights 3/5m, yaw offsets -45..45 in 22.5-degree
  steps, pitches -20/-10. No hidden full 0.5m candidate-domain claim.
- Existing forbidden intervals, one sensor per XY pole, and 4m minimum spacing
  remain hard constraints. Critical targets are within 20m of A's junction or
  within 12m station distance of B's module interfaces.
- Four constraint profiles: base, cost, cost+critical60, cost+critical80.
  This phase reports overlap but does not impose a new overlap floor. Existing
  nonzero overlap constraints remain supported by the shared solver and tests.
- 16 case workers, 64 map/scene/profile conditions, 184 full-domain reference
  rows, 1,288 online rows. These are not 1,288 independent road samples.

## Objective and Hard Constraints

Let x_i select a candidate, a_ij be its physical visibility, and T_c be the
geometry-defined critical target subset. All targets have equal weight.

```text
d_j = sum_i a_ij x_i
y_j = 1[d_j >= 1]
maximize sum_j y_j
sum_i x_i <= K
sum_i c_i x_i <= B_K             # cost profiles only
sum_(j in T_c) y_j >= ceil(tau |T_c|)
sum_(i on pole s) x_i <= 1
x_i + x_l <= 1                  # poles closer than 4m
```

The implementation uses d_j >= y_j and d_j <= K y_j with binary x/y.
Identical incidence columns are compressed with multiplicity weights, keeping
critical flags separate. This preserves all target constraints exactly.

```text
c_i = round(1 + 0.10 max(z_i - 3, 0) + 0.008 distance(xy_i, H), 6)
B_K = 1.25 K
tau = 0 / 0 / 0.60 / 0.80 for the four profiles
```

These are synthetic relative units, NOT market prices or a deployment standard.
A's service anchors are the nearest admissible pole sites to (0.15L, y_junction)
and (0.85L, y_junction). B's anchors are the nearest pole sites to centerline
stations 0,160,320,... plus the endpoint. Nearest Euclidean distance is a cost
proxy, NOT cable routing, service capacity, or communication reachability.
Site geometry and costs are identical across methods and scene levels. No
coverage result is used to set anchors, prices, budgets, or thresholds.

## Methods and Information Boundary

Retrieve three compatible sources with the existing geometric distance rule.
Compare library_prefix, library_diverse, geometry_union, geometry_portfolio,
and random(seed=0,1,2). The library is not regenerated for these constraints.
The geometry portfolio's optical surrogate obeys the same cost/critical profile.
All methods use the SAME physical constrained MILP for final subset selection.

Requested Q=256; actual Q=min(256, transferred library pool size), shared by ALL
methods in that case. Record both. There is one Q level, no larger-Q regression
or cross-Q warm-start comparison in this study. All plans are frozen before any
target physical query. Methods have fresh visibility memoization; only geometry
and their own evaluated poses are available online. The full-domain reference
runs afterwards without reading online solutions as warm starts.

## Solver and Resource Limits

- Gurobi, 1 thread per worker; 2 workers; 24GB solver soft limit each.
- Per K: online 300s, full reference 1800s, optical surrogate 15s.
- MIPGap=0; optional early termination requires a VALID gap <=1%, at least
  300 solver seconds (15 for surrogate), and 120s without improvement.
  Plateau alone never certifies near-optimality.
- Solver limits exclude geometry, query planning, model construction and export.
- Supervisor guards: 48GiB available RAM before launch, individual RSS 40GiB,
  combined RSS 80GiB, 12GiB disk reserve; job 48h, total 168h hard limits.
  Soft memory is not a whole-system guarantee. Resource failures stop remaining
  jobs and preserve completed files. No automatic resume or silent retry.
- Nominal sum of online/reference solver caps: 199.33 worker-hours, excluding
  model construction and surrogate solves. Actual duration may be much shorter;
  there is no imposed minimum training time and this is optimization, not RL.

## Validation and Interpretation

Each case checks every queried mask against the newly built full archive and
reraycasts every selected physical pose. Every feasible deployment is rechecked
for cost, critical coverage, spacing, pole capacity, and K. Source/code/spec
hashes and immutable shortlist hashes are verified. Source physical validation
was already completed; this run verifies and copies source artifacts unchanged.

Report feasible rate, proven-infeasible and no-feasible-found counts separately.
A shortlist proven infeasible is NOT proof of full-domain infeasibility. Unknown
or infeasible results are null, never zero-coverage observations. Compare quality
on paired feasible conditions and also report how many conditions were excluded.
Average random seeds within each condition before aggregating maps. Repeated K
and obstacle levels from one map are dependent observations.

Report total coverage, critical coverage, overlap (density>=2 / all targets and
also / covered targets), actual sensor count, cost, feasible-reference difference,
upper-bound loss, solver status/gap, rays, planning/query/selection times and
cumulative K-frontier latency. K levels share preparation, so cumulative latency
is not standalone latency for each K. Hardware is shared, not an isolated timing
benchmark. Offline library construction/amortization is separate.

This is a constraint-transfer pilot, not a global-optimality proof, unseen-road
generalization result, N-1 robustness result, or minimum sensor-count certificate.
No parameter is adjusted after observing the current experiment's performance.

## Commands

```bash
# Read-only plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_budget_study.py \
  --spec configs/library_lowk_cost_20261124.json

# Tiny pipeline verification only, not performance evidence; use a fresh root.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_budget_study.py \
  --smoke --output results/library_lowk_cost_smoke_20261124_v2 --execute

# Formal execution; use a fresh root on subsequent runs.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_budget_study.py \
  --spec configs/library_lowk_cost_20261124.json \
  --output results/library_lowk_cost_20261124 --execute --detach
```

Lifecycle only is monitored during execution. Performance logs and method
comparisons are inspected after completion. Root `comparison.csv` is assembled
only after all 16 case workers finish. Per-case artifacts include cost/hubs,
transfers, frozen plans, queried arrays, time/bound traces and direct validation.

## Preflight Record

118 focused/regression tests passed, including independent exhaustive subset
enumeration with joint cost/critical/overlap constraints. Syntax checks, the new
formal dry-run and all three existing method CLI dry-runs passed.
`results/library_lowk_cost_smoke_20261124_v2` completed 2/2 cases and 80/80 rows,
with direct-raycast validation and matching frozen implementation hashes.
This is functional verification, not performance evidence.

The first smoke attempt used an overly coarse 20m installation grid, which
removed a branch/side required by source transfer. It failed explicitly and is
preserved at `results/library_lowk_cost_smoke_20261124`. The final smoke keeps
the same 4m installation grid and height/orientation modes as the formal study,
reducing only target sampling and solve budgets. A regression test now protects
these modes. No formal performance results were used to make this correction.
