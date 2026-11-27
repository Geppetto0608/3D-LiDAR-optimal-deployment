# Fixed-Q Library Mechanism Ablation

## Material Passport

- Date: 2026-11-27.
- Mode: implementation and functional verification, not a new performance claim.
- Inputs: the completed 24-source `results/library_rsu_stress_20261123` library.
- Code/config: `library_mechanism_study.py`, `configs/library_mechanism_20261127.json`.
- Formal results: **TBD; not launched**. Existing experiments remain unchanged.
- Scientific question: at a fixed target pose-query budget, which parts of a
  library-informed candidate proposal contribute useful placements?

## What Is Isolated

The previous low-K runner used `q = min(requested_Q, library_pool_size)` for all
methods within a condition. This was matched within a condition, but changing
library size could also change q. It cannot isolate a library-size effect.

This runner holds Q=256 for every arm, scene, profile and target. A target with
fewer than Q eligible poses is rejected, not silently assigned a smaller Q.
All methods use the same poses, targets, optics, static occluders, installation
rules, costs, critical-coverage floors and final constrained MILP as the low-K
study. No source reoptimization, learned policy, detector or new physical model
is added. Both A (topology) and B (component/corridor) tracks remain separate.

| Factor/control | Values |
|---|---|
| Retrieved sources | 1, 3, 6, same topology for A; compatible B corridors |
| Layouts per source profile/K | incumbent only (1); up to 12 including incumbent |
| Main shortlist | optical union/double-gain diverse selection |
| Prefix control | nearest 3 sources, up to 12 layouts, stored/interleaved order |
| Source-search control | 3 uniformly sampled compatible sources, seeds 0/1/2 |
| Non-library controls | geometry union; geometry-portfolio MILP; random poses, seeds 0/1/2 |
| Target geometries/scenes | the same 8 development geometries, mixed/dense |
| K and constraint profiles | unchanged low-K base/cost/critical60/critical80 |

There are 15 arms per profile: six nearest-source factorial variants, one
prefix, three random-source trials, two geometry controls and three random-pose
trials. Expected output is 2,760 online rows plus 184 reference rows in 16 case
workers. Random-source seeds and random-pose seeds are different experiments,
not extra independent roads. Six sources exhaust the A topology-compatible
catalog but are only half of B's catalog.

"Incumbent only" means the recorded feasible `selected_ids` for **every stored
source profile/K**, not one layout across the entire library. Alternatives are
deduplicated within a profile/K and capped including the incumbent. They are
not exact top-N solutions, and not every row has 12. Profiles and K values are
held fixed while the number of stored layouts admitted is varied.

## Explicit Hybrid, Not Hidden Padding

Let E be the eligible target pose set, L_v the geometry-transferred source
pool for variant v, and G_Q an ordered, Q-pose geometry-union shortlist. No
physical target mask is used in constructing either pool.

```text
R_v = prefix_or_diverse(L_v, min(Q, |L_v|))
H_v = first (Q - |R_v|) distinct entries of G_Q not in R_v
C_v = R_v union H_v
|C_v| = Q
```

Since G_Q has Q distinct poses, it contains enough poses outside any smaller
R_v. Empty L_v becomes an explicitly labeled 100% geometry fallback. Missing
installation-side mappings still fail instead of manufacturing a source pose.

Every plan records `library_ids`, `geometry_fill_ids`, `library_query_count`,
`geometry_fill_count`, `empty_library_fallback`, pool size and source identities.
Controls have zero *fill* because their complete shortlist is their own method,
not a library pool being supplemented. The raw CSV carries the count metadata.

**Estimand:** the benefit of a library-informed **hybrid** at fixed Q, including
replacement of geometry-fill proposals as the library grows. This does not
isolate "pure library only" at Q=256 when a pool has fewer than 256 poses.
Report fill fractions with quality; do not relabel this as pure-library proof.
If two variants produce the same query set, that is a zero proposal effect,
even if time-limited solver paths happen to return different incumbents.
Query-set hashes and distinct queried XYZ counts are recorded; paired tables
flag conditions in which every trial on both sides queried exactly the same set.

Q counts distinct **poses**, not XYZ sites. Several orientations can share
line-of-sight work, so fixed Q does not imply equal rays, wall time or FLOPs.
Actual ray counts, visibility time, planning time and solve time remain separate.

## Transfer and Information Boundary

Nearest retrieval keeps the established geometry-distance rule. Random retrieval
uniformly samples without replacement from compatible sources, sorting names
before the seeded permutation so directory order cannot change the draw.
Neither selection sees physical target coverage or reference solutions.

A uses branch/side-relative installation-segment fraction and relative pose.
B maps station fractions into each target module with side, height and relative
orientation held compatible. Different source layouts are not coordinate-averaged.
Transferred source orders are interleaved and duplicate target poses removed.

All profiles' plans and transfer records are written before the first target
physical query. Every arm has a fresh visibility provider and local LOS cache.
Only after all online arms finish does the full-domain reference query/solve run.
The reference never receives online deployments as warm starts. Source masks
are retained for provenance but not transferred to target visibility.

Online and reference mask agreement, a fresh direct raycast of selected poses,
common constraint evaluation and query-plan hashes are checked before a case is
marked complete. These reuse the same physical implementation; they do not
validate a real sensor or an independent simulator.

## Prospective Reserved Split

The config embeds 24 source descriptors, eight known development descriptors,
and eight prospective reserved descriptors:

- A: T/cross at 140/200 m, widths 9/13 m, centers (.38,.59)/(.61,.39).
- B: lengths 200/280/360/520 m, widths 9/13 m, turns 22.5/37.5/67.5/82.5 degrees.
- Reserved descriptor seeds: 7000..7007.

Names, geometry identity and source/development/reserved obstacle seeds must be
disjoint across partitions. Geometry identity ignores names/seeds; for B, changing
only module count also does not create a new road geometry. The run snapshot
hashes the entire config, including the reserved partition.

This runner accepts **development targets only**. The reserved descriptors are
not constructed, raycast or solved in dry-runs, unit tests or smoke execution.
Final evaluation needs a separately frozen protocol after development choices
are settled; editing this runner to use reserved maps would invalidate that claim.

This is a prospective local reservation, not proof that every historical run has
never encountered similar roads, nor a registered external benchmark. The
geometry groups differ, but topology families remain T/cross/corridor: no unseen
topology generalization claim. The current B object generator is deterministic
in station placement and does not use its seed to vary positions; changing B
seed alone is not an independent occlusion replicate. Real irregular road-network
validation and independently varied blocker configurations remain future work.

## Optimization and Reporting

The final objective is still unweighted union coverage. At-most-K, pole capacity,
4 m separation, forbidden positions, the synthetic cost cap 1.25K and optional
critical floors 60%/80% are unchanged. Required overlap is zero in this low-K
study, but overlap/double-coverage metrics are reported. No operational cost or
service threshold is being asserted.

MILP caps per K: online 300 s, full reference 1,800 s, optical portfolio 15 s;
one solver thread, 24 GB soft memory. Early stopping still requires a finite
bound-derived gap <=1%, minimum elapsed time and stagnation. Flat coverage alone
never certifies near-optimality. Queried-pool optimality is not full-domain
optimality. Full reference bound/gap and loss intervals remain in raw rows.

The sum of solve caps is about 322.77 hours, not a runtime forecast or required
duration. Two case workers, 240-hour launcher / 48-hour worker hard limits are
safety bounds. Whole-study completion before those guards is not guaranteed.

`comparison.csv` contains costs, critical coverage, overlap, feasibility, bounds,
timing, rays, pool/fill counts and factor labels. `paired_effects.csv` contains:

- source-count increments at each alternative count;
- alternative-count increments at each source count;
- the nearest-3/12 diverse anchor against prefix, random source and controls.

For stochastic controls, average requested seeds **within a condition first**.
If any requested seed lacks a feasible solution, the complete paired coverage
difference is missing, with feasibility flags retained; it is never zero-filled.
`ablation_summary.csv` reports paired counts and map-balanced means over those
common-feasible conditions. It is conditional descriptive analysis, not an
unconditional success rate or significance test. Check infeasible/unknown rows
separately, and do not count K/profile/scene rows as independent map samples.
The report rejects duplicate/missing conditions/seeds and unequal Q.

`online_all_k_sec` describes one whole frontier; do not sum its repeated K rows.
Planning components reused in a worker are charged per hypothetical standalone
arm. Geometry mask computation is charged when needed for diversity or fill.
Shared server load, OS caching, imports, library integrity audits, final direct
validation and export prevent interpreting this as a controlled cold-start
speedup. Offline source build cost is not included in online latency.

## Commands

```bash
# Read-only plan; source hash checks and development geometry construction only.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_mechanism_study.py \
  --spec configs/library_mechanism_20261127.json

# Functional check, not performance evidence. Use a new root each time.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_mechanism_study.py --smoke \
  --output results/library_mechanism_smoke_repeat --execute

# Formal development ablation. NOT run as part of implementation verification.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_mechanism_study.py \
  --spec configs/library_mechanism_20261127.json \
  --output results/library_mechanism_development_repeat --execute --detach

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_mechanism.py
```

Existing output roots are refused. The completed source library is required;
ignored result files do not accompany a repository clone. No solver fallback or
automatic retry/resume is added. Frozen code/config/source-library hashes,
hardware/packages, per-case logs and supervisor status follow the existing runner.
Final results must be analyzed after completion; no repeated log analysis is needed.

## Verification Status

Verification completed on 2026-11-27:

- 135 tests passed, including ten new mechanism tests and the existing low-K,
  result-audit, geometry/visibility, candidate-control and pipeline regressions.
- Syntax checks, the new formal dry-run, and the legacy MILP/GREEDY/MAPPO
  dry-runs passed. Dry-runs do not query physical visibility or solve.
- `results/library_mechanism_smoke_20261127`: 2/2 A/B cases complete, 88/88 online
  rows feasible, eight full-domain reference rows; launcher wall time 42.03 s.
- All 44 query plans have exactly Q=32 distinct poses. Library geometry-fill
  counts range 0..18 on A and 0..8 on B; no pool-dependent budget shrink occurred.
- Each case passed fresh direct-raycast validation and common constraint checks.
  A post-run read-only check also matched all queried masks against the saved
  full candidate archive and verified frozen code/config/plan hashes.
- Both jobs used existing development maps; none used reserved descriptors.
- `paired_effects.csv`, `ablation_summary.csv`, `report_status.json` completed.

Exact functional execution command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_mechanism_study.py --smoke \
  --output results/library_mechanism_smoke_20261127 --execute
```

The root now exists; use a new name for a repeat. Formal ablation and reserved-map
performance remain **TBD**. Smoke uses Q=32, K=2/3, base/critical60 profiles,
one random seed and deliberately short solve limits. These results establish
workflow/constraint consistency, not library advantage or near-optimal quality.

Smoke runs one worker with a 4 GiB runtime disk reserve (8 GiB start preflight),
while the formal experiment retains the 12 GiB reserve (16 GiB start preflight).
This distinction does not permit overwriting or deleting previous results.
An initial attempt stopped before output creation at the formal disk preflight;
available disk was about 12 GB. Only the small smoke reserve was changed, and
no existing result was removed. Formal execution must first meet its own guard.
