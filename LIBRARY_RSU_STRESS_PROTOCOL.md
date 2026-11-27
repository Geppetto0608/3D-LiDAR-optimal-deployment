# RSU Coverage Library Stress Benchmark (2026-11-23)

## Question and Scope

Can transferred source-library proposals retain more feasible geometric coverage
than geometry-only or random proposals when maps become larger, static occlusion
increases, and mandatory double-observation constraints are introduced?

This is **constrained deployment optimization, not MAPPO training**. A/B denote
different library transfer tracks, not competing methods on the same geometry.
Existing code, old libraries, and previous results are preserved. The runner is
`library_rsu_stress.py`; the frozen input is
`configs/library_rsu_stress_20261123.json`. No detector, tracking accuracy,
radio-coverage model, learned policy, arbitrary city road graph, or continuous
global-optimum claim is included. Results of the full run are **TBD** until its
manifest is complete. A 12-hour minimum duration is not artificially imposed.

## Physical Motivation, Not Deployment Standards

The [FHWA Traffic Monitoring Guide](https://www.fhwa.dot.gov/policyinformation/tmguide/tmg_2022/traffic-data-collection.cfm)
and [Traffic Detector Handbook](https://www.fhwa.dot.gov/publications/research/operations/its/06139/chapt5d.cfm)
motivate explicitly examining obstruction and sensor siting rather than assuming
ideal sensing everywhere. They do **not** prescribe this study's LiDAR overlap
ratios, separation distance, or obstacle dimensions. Those are illustrative,
predeclared stress settings, not validated operational RSU requirements.

## Frozen Suite

| Axis | Setting |
|---|---|
| A source library | T/CROSS x square side 100/160m x road width 8/12/16m: 12 maps |
| B source library | Corridor length 80/120m x width 8/12/16m x bend 0/45 degrees: 12 maps |
| A targets | T/CROSS at 120x120m and 180x180m, width 10/14m, shifted junctions: 4 maps |
| B targets | Length 160/240/320/480m, width 10/14/10/14m, bend 15/30/60/75 degrees: 4 maps |
| Source environment | Fixed mixed-occlusion scene per geometry |
| Target environments | clear, mixed, dense: 24 geometry/scene jobs |
| Requirements | union and redundancy: 48 logical conditions |
| Source sensor budgets | Union K=1..8; redundancy K=4,8 |
| Target sensor budgets | K=4,8,12, always at-most-K |
| Library retrieval | 3 geometry-nearest compatible sources; no target coverage in retrieval |
| Stored alternatives | Up to 12 feasible layouts per source/profile/K, not guaranteed 12 distinct optima |
| Target query budgets | 128/256 orientation-specific poses, capped by actual transferred pool size |
| Random repeats | Seeds 0/1/2; deterministic methods are not counted as replicated trials |
| Planned comparison rows | 2,016; failure/infeasibility is never filled with zero coverage |

There are eight target geometries, not 2,016 independent maps. Scene, constraint,
budget, method, and random-seed rows are paired/dependent observations. The suite
contains one fixed obstacle arrangement per geometry, not a Monte Carlo estimate
over traffic realizations or a statistically conclusive deployment study.

A uses the existing MapGeometry with explicit enlarged dimensions, 0.5m target
XY cells, and targets at z=1/1.5/2m. B uses 1m station/lateral target sampling at
the same heights. A and B percentages are not compared as matched measurements.
B is still a straight/arc/straight corridor with a single circular bend, divided
into 2/3/4/6 transfer modules. It is not a multi-intersection urban road network.

Both tracks use nominal 4m candidate station/bin spacing, z=3/5m,
road-normal-relative yaw=-45/-22.5/0/22.5/45 degrees, pitch=-20/-10 degrees.
A retains the established first-installable-cell-per-bin rule; the candidate
space is **not every 0.5m position or every orientation**. There is no top-10
visibility pruning and no pose-mask deduplication. The sensor profile remains
360-degree horizontal FOV, 26.9-degree vertical FOV, and 30m maximum range.

Geometry-only preflight counts: source pools 740..2,840 poses; target pools
1,480..4,460 poses. Target counts are 4,458..54,852, depending on track/geometry.
The largest target dense working mask is about 0.161 GiB, not total solver RSS.

## Static Objects and Installation Restrictions

Fixed, seeded opaque boxes represent vehicle-sized or roadside-panel-sized
obstructions: 4.5x1.8x1.8m, 10x2.5x3.5m, and 6x0.25x2.5m. Their bottoms are 0.1m
above the reference ground. Counts scale with source/target length, with 6..24
potential objects on target maps. Mixed uses the first half of a fixed object
list; dense uses all. Existing roadside buildings/walls remain in every scene.

All potential object XY footprints are removed from the target support in
**every** scene, including clear. Candidate collisions are likewise removed in
every scene. Clear/mixed/dense therefore compare the same targets and candidate
positions; there is no artificial target removal only in the obstructed case.
This is a deliberately fixed free-space-support experiment, not coverage of
occupied vehicle interiors or a dynamic traffic simulation. Removed counts,
seeds, dimensions, and object-generation code are preserved.

A additionally excludes east-branch installation fraction u in [0.32,0.43]
and west/left u in [0.70,0.80]. B excludes station fraction [0.18,0.23] on both
sides and [0.66,0.70] on its positive-lateral side. These are synthetic
unavailable-installation intervals, not surveyed utility or legal constraints.
At most one sensor may use a pole (identical XY), even at different heights or
orientations. Different selected poles must be at least 4m apart in Euclidean XY.

## Exact Finite Objective and Constraints

For candidate i and target j, a_ij is the common physical LOS/range/FOV mask.
For binary candidate selections x_i and an at-most-K budget:

```text
d_j = sum_i a_ij*x_i
y_j = 1[d_j >= 1]
z_j = 1[d_j >= 2]
maximize F = sum_j y_j
sum_i x_i <= K
d_j >= y_j; d_j <= K*y_j
d_j >= 2*z_j; d_j <= 1 + (K-1)*z_j
sum_(i at pole p) x_i <= 1
sum_(i at p or q) x_i <= 1, if distance(p,q) < 4m (numerical tolerance 1e-7)
```

The union profile adds no double-observation minimum. The redundancy profile adds:

```text
sum_j z_j >= ceil(0.10*N)
sum_(j in critical region) z_j >= ceil(0.30*N_critical)
```

A critical region is within 20m XY of the junction. B critical regions are
within 12m station distance of the module interfaces. Double observation counts
distinct installed sensors, not returns or distinct sight angles. Neither this
ratio nor double observation proves detection continuity, robustness to every
failure, adequate point density, or independent sensing directions.

Targets with identical candidate-incidence columns are losslessly grouped,
**separately for critical and ordinary targets**, and exact multiplicities weight
the objective/constraints. No target or candidate subsampling occurs inside the
solver. Double-observation variables are omitted when both overlap minima are
zero; reported overlap is still recomputed from actual density. This formulation
is distinct from the legacy union-only coverage cuts.

## Compared Proposal Methods

Every proposed pool is physically evaluated and selected by the same constrained
MILP; methods differ in which positions they propose, not in their final solver.

1. `library_prefix`: geometry retrieval, road-relative transfer, historical
   round-robin prefix. It deliberately preserves the truncation rule as an ablation.
2. `library_diverse`: the same transferred union, ranked with a target optical
   union/double-coverage surrogate. After optical saturation, spatial separation
   fills the remaining query slots. Actual occlusion masks are not used to rank.
3. `geometry_union`: no library; optical union greedy over the whole valid domain,
   followed by the same spatial fill after saturation.
4. `geometry_portfolio`: no library; multiple constrained optical MILP layouts
   across target K, then geometry-union fill. Optical masks ignore all occlusion.
5. `random`: no library; seeded random poses without replacement, three seeds.

Requested Q is capped by the available library pool, **without hidden padding**.
Every method in the matched condition uses that actual Q; actual_q and pool_count
are stored. Equal pose-query counts do not imply equal unique XYZ queries, ray
counts, planning cost, or wall time. Each arm uses fresh LOS memoization and
independent physical evaluations. Source masks are not transferred as target masks.
Both Q values use prefixes of a single maximum-Q plan. The full planning cost is
charged to each Q row; the smaller-Q time is not the latency of an independently
optimized shorter planning pass. Duplicate pools after capping remain explicitly
labelled by requested_q/actual_q and are not independent samples.

A transfer preserves branch/side, normalized installation station u, height, and
relative orientation, then projects to valid target candidates. B transfers each
source layout to each target module by normalized station/side/orientation.
Collisions are deduplicated as candidate proposals and recorded. A mapped layout
is not assumed feasible or optimal; the final target selection is recomputed.
Queries are frozen and hashed before any target physical evaluation. The full
target reference is evaluated afterward and never supplies online warm starts.

## Solver Limits and Certificates

Gurobi is required; there is no silent backend fallback. Each source and full
reference K gets at most 1,800 solver seconds, each queried-pool K 300 seconds,
and each geometry-only surrogate K 15 seconds. Model construction, compression,
heuristics, visibility, serialization, and validation are outside that solver cap.
One solver thread and 12GB Gurobi soft memory are used per worker.

Exact termination may finish early. Other gap-based termination requires all of:
at least 300 solver seconds, at least 120 seconds without improving the feasible
incumbent, and certified relative gap <=1%. Stagnation alone is never evidence of
near-optimality. For the short surrogate run these conditions normally cannot
trigger. `near_optimal_certified` denotes the saved final bound, not elapsed time.

The [Gurobi solution pool documentation](https://docs.gurobi.com/projects/optimizer/en/current/features/solutionpool.html)
distinguishes heuristic alternative search from exact n-best enumeration. This
runner uses PoolSearchMode=1 and does **not** certify the stored alternative list
as the top 12 layouts. Each stored layout is independently feasible. During
pool-search callbacks, a bound below a verified incumbent is not used for a gap
claim; a safe all-candidate union bound replaces it and the source is recorded.
The final solver bound is checked against the verified feasible objective.

For maximization, L is a feasible objective and U a valid upper bound. The saved
gap is (U-L)/L; zero coverage and missing feasible solutions are handled explicitly.
Certificates cover only the supplied discrete pool. An exact queried-pool solve
does not certify global optimality over the full target domain. Reference loss
is reported as [max(0,L_ref-C), max(0,U_ref-C)] in percentage points. Missing
feasible deployments have no numeric coverage/loss, with `infeasible` (proved)
and `no_feasible_found` (not proved) kept distinct.

## Runtime and Resource Boundaries

Four independent CPU workers run sources first, then target jobs. Processes,
resource limits, failures, and completion are supervised without parsing live
performance logs. Every source/profile/K and target arm writes quality-time
CSV/JSON, overlap, selected counts, status/bounds, and timings. Root aggregation
occurs only after all target jobs finish. Interpretation waits until completion.

Reported online time sums geometry/scene preparation, method-specific planning,
physical queries, and optimization. All-K time shares preprocessing across K;
through-K time is cumulative in ascending K, **not a fresh single-K latency**.
Process startup/top-level imports, final independent validation, and archive
serialization are excluded from that metric and included in job wall time.
The full reference and source construction times are reported separately.
Parallel jobs share the server and OS caches, and deterministic methods are not
timing-repeated. These measurements are not a controlled hardware speedup claim.

The configured sum of solver caps is 360 serial worker-hours, or about 90 hours
at ideal four-way concurrency before overhead; this is an upper-budget sum,
**not a runtime forecast**. Easy/certified problems can finish much earlier.
Supervisor hard limits: 120h total, 18h/job, 24GiB/job RSS, 80GiB sum of own worker
RSS. New jobs require >=32GiB available RAM and >=8GiB free disk; critically low
system memory stops the experiment. These sampled guards and Gurobi soft limits
are not a cgroup-enforced guarantee against OOM. Interrupted artifacts are kept;
automatic optimizer-checkpoint resume is not implemented.

## Reproduction and Outputs

```bash
# Plan only: no output, raycasting, or optimization.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_rsu_stress.py \
  --spec configs/library_rsu_stress_20261123.json

# New result root required; returns after detached supervisor startup.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_rsu_stress.py \
  --spec configs/library_rsu_stress_20261123.json \
  --output results/library_rsu_stress_20261123 --execute --detach

# Optional human progress view; the assistant does not repeatedly tail it.
tail -F results/library_rsu_stress_20261123/launcher.log
```

```text
manifest.json / spec.json / launch.json / packages.txt / launcher.log
source/                           frozen code, test, protocol, hashes
logs/ / job_status/                worker lifecycle and validation records
library/<source>/                 layouts.json, candidates.npz, result.json
  union/ redundancy/              frontier.json, progress.jsonl, quality_time.csv
cases/<geometry>_<scene>/         transfer.json, query_plans.json, candidates.npz
  <profile>_q<Q>_<method>[_s<seed>]/ queried.npz, result.json, frontier/quality trace
  reference_<profile>/            independent full-domain finite reference
  comparison.csv / result.json    final condition rows and direct validation
comparison.csv                    complete 2,016-row aggregate, only at the end
```

Raw solver text is disabled; structured progress files contain incumbent/bound
curves. A `launcher.log` may be quiet while a long source worker is solving.
Saved arrays and code are ignored result artifacts, not included by repository clone.

## Verification

New tests check 18 small constrained MILP problems against exhaustive enumeration,
one-pole redundancy rejection, empty-pool semantics, critical-group compression,
geometry-only dry-run guards, scene-domain invariance, fast/direct mask equality,
large-map transfer, shortlist nesting, and solution-pool bound regression.
The 10 tests and the existing 7 pipeline tests pass; existing MILP/GREEDY/MAPPO
CLI dry-runs pass. The corrected frozen smoke execution
`results/library_rsu_stress_smoke_20261123_v4` completed 4 source + 4 case jobs,
160 rows, all eight direct-validation records. The failed initial smoke is
preserved separately; it exposed the callback-bound issue described above.
These are functional checks, not evidence of large-map superiority.

Full-resolution preflight also constructed the 180m CROSS and 480m corridor,
queried all 3,160/4,460 poses, matched 40 fresh direct pose masks on each, and
exercised a one-second constrained K=4 solve. The compressed incidence sizes
were 22,072,895 and 7,410,896 edges, below the explicit 80M-edge safety guard.
This is a memory/model integration check, not a performance comparison.
