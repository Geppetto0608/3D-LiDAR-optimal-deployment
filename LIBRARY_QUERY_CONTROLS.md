# A/B Equal-Query Development Check

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run, with explicitly authorized implementation of controls
- Origin Date: 2026-11-18, Asia/Seoul
- Verification Status: protocol fixed before new control outcomes
- Version Label: library_matched_queries_v1

## Question And Scope

Does the existing transferred library pool retain better target coverage than
source-free selection at the **same number of distinct physical pose queries**?
This is not a timing superiority test, novelty claim, or untouched generalization
test. The four targets are already inspected development cases. Source, target,
candidate axes, physical model and K values stay unchanged from the previous
A/B target smoke. No new learning, operational constraints or library tuning.

| Case | Frozen source | Full poses | Matched queries | K |
|---|---|---:|---:|---|
| a_w9 | library_ab_smoke_20261118/a_v2 | 30 | 11 | 1,2,3 |
| a_w12 | library_ab_smoke_20261118/a_v2 | 25 | 8 | 1,2,3 |
| b_turn45 | library_ab_smoke_20261118/b | 48 | 20 | 1,2,3 |
| b_turn60 | library_ab_smoke_20261118/b | 48 | 20 | 1,2,3 |

All source directories are under `results/`. The matched budget is the cardinality
of the geometry-transferred pool, fixed without target coverage. Controls receive
only the common pose grid, target coordinates, sensor profile, K and that scalar
budget. They receive no library geometry, poses, masks, scores, candidate IDs or
target reference. Sharing the declared physical problem is not source transfer.

## Fixed Arms

1. **library:** unchanged transferred candidate pool, preserving feasible source
   layouts after target reevaluation. The downstream solver may recombine poses.
2. **geometry_single:** highest unoccluded range/FOV target counts; candidate ID
   breaks ties. No raycast or source information in planning.
3. **geometry_union:** sequential largest additional unoccluded target count;
   ties use individual count then ID. Continue to the full query budget even
   after predicted union saturates, without repeated candidate IDs.
4. **geometry_portfolio:** exact finite-grid optimization of unoccluded coverage
   at each K, retaining up to the query budget's number of alternative layouts.
   Scan alternative rank first, increasing K next, then ordered candidate IDs,
   adding unseen poses until the budget is full. If needed, fill with the
   geometry_union order. This is a deliberately stronger source-free control,
   not a newly proposed research method or a free computation.
5. **random:** uniform candidate sampling without replacement for each fixed
   seed 0..29. All seeds are retained, not only the best seed.

Unoccluded predictions use the existing optical convention and ignore obstacles;
they are not actual coverage. Geometry planners inspect all pose-target pairs,
and the portfolio also enumerates predicted subsets. Record that work and its
planning time. Do not count it as zero merely because it makes no physical ray
queries. Random uses no optical predictions.

## Execution Boundary

All query plans are fixed before the first physical provider is created. Each
arm gets a new provider and queries exactly q distinct poses on the actual target
mesh. No cross-arm mask cache is shared. K1/2/3 reuse that arm's q masks.
Every arm uses the same exact subset backend and tie rules within its queried
pool. Exact enumeration must fit the declared guard; otherwise the study fails
rather than mixing approximate and exact selection.

Only after every arm's deployment has been fixed is the full target reference
raycast and solved. The full reference receives no source/arm warm start.
Compare arm masks with the reference masks at the same IDs, then verify all
saved deployments using a fresh provider. Freeze source/inputs/implementation,
check hashes before and after, use new output directories, and preserve old runs.
Workers are sequential with a 600-second hard timeout each. Do not retry failures
silently or tune methods after looking at these results. Read logs after completion.

There are 34 arms per map (library + three geometry controls + 30 random draws),
136 map/arm evaluations and 408 K rows, plus 12 full-reference K rows.
There are only four development maps, not 136 independent maps. Deterministic
methods are not copied 30 times and treated as new observations.

## Metrics And Interpretation

For actual target visibility sets A_i, use the existing unweighted union:

```text
f(S) = |union(i in S) A_i|, |S| <= K
L_method(K) = max f(S), S subset queried_pool_method
regret_count = f*_full(K) - L_method(K)
delta_pp = 100 * (L_library(K) - L_control(K)) / target_count
```

Report all case/K scores, full-optimum match counts, exact candidate query counts,
actual submitted rays, planner work and separate timing components. Random gets
mean, sample SD (ddof=1), p10/median/p90 using linear quantiles, and library-vs-seed
win/tie/loss counts. These are conditional random-selection distributions on the
same maps, not confidence intervals for unseen roads. Report the best random
draw only as a distribution diagnostic, never as an available online method.

Compare against every predeclared control, not whichever is weakest. If a stronger
geometry control matches or beats the library, state that limitation. A win over
random alone does not establish source reuse value. Any equal-weight mean across
case/K is descriptive; it is not target-count-weighted and maps sharing geometry
or sources are not independent. No p-values or publication-readiness conclusion.

Equal pose query counts do not equalize actual rays (the optical filter varies),
CPU, offline preparation or end-to-end latency. These workers share mesh/imports
between arms and run sequentially on a shared machine. Source loading/mapping
is recorded once separately, and source construction remains an additional cost.
No speedup claim or full runtime amortization is possible from this run.

## Reproduction

Add `--control-seeds 0 1 ... 29` to `library_transfer_study.py` for the opt-in
study. The literal ellipsis is not a CLI argument; the README provides complete
commands. Omitting the flag preserves the original single-library evaluation.
Without `--execute`, validate geometry/budget plans without ranking, raycasts,
subset optimization or writes. New schema: `library_matched_query_evaluation_v1`.
`library_query_report.py` aggregates completed runs, checks masks/poses and can
compare old library/full outputs only after all new selections are fixed.
The original results remain under `results/library_ab_transfer_20261118/`.
