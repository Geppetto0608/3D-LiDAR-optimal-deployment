# Fixed-Library Occlusion Stress Protocol

Frozen development protocol, 2026-11-22. This extends the matched-query
comparison only by adding opaque geometry to the target scene. It does not
claim actual road infrastructure compliance or object detection performance.

## Design

- Keep A `library_ab_smoke_20261118/a_v2` and B `library_ab_smoke_20261118/b`
  unchanged; no new source optimization, library expansion, RL, LS or costs.
- Reuse four previously inspected development maps. Each has a fresh clear
  evaluation and an occluded evaluation: eight sequential target jobs.
- K=1,2,3; library, geometry_single, geometry_union, geometry_portfolio and
  random seeds 0..29. Same at-most-K exact finite-subset selector in every arm.
- Query budgets remain A/w9=11, A/w12=8, B/45=20, B/60=20. Neither candidate
  nor target arrays/order nor the transferred pool changes between conditions.
- All 34 query plans precede target raycasting. A method sees physical masks
  only for its queried poses. All decisions precede full-reference evaluation.
  Geometry controls deliberately remain LOS-free controls, not obstacle-aware
  planners. This is a scoped comparison, not a strongest-baseline claim.
- Clear means no *additional* screens; preexisting buildings remain.

## Geometry-Only Screen Recipe

Two opaque boxes per target, thickness 0.2 m, height 4.5 m from ground,
span 70% of road width, centered on the road and aligned transverse to it.
A places boxes 20 m east/west of the junction on the horizontal road.
B places boxes at station fractions 0.25 and 0.75, oriented by road tangent.
These are synthetic stationary screens, not traffic barriers with validated
dimensions, vehicle trajectories, dynamic occlusion, or siting restrictions.

All boxes are resolved and frozen before the first physical query. Any box
containing a sensor candidate or evaluation point is rejected, not repaired by
dropping that point or candidate. No outcome-driven relocation is permitted.
Occluders affect Euclidean mesh raycasting only; no mask approximation or wrap.

## Validation and Reporting

Each target run checks queried masks against the full archive and freshly
raycasts selected poses. After all jobs, paired validation checks identical
domains and query IDs, and that added geometry cannot create new visibility.
Removed candidate-target edges and affected candidate counts are recorded;
zero-effect cases remain in the report rather than being silently replaced.
Source libraries, target inputs, resolved boxes and code are copied and hashed.
Existing results are never overwritten. Failure stops the batch with logs.

Results describe the declared coarse finite domains only. Four paired geometry
cases are not eight independent maps, and random seeds are not independent
road networks. Equal queried poses do not equal rays or end-to-end runtime.
No controlled speedup, real-world generalization, or method superiority claim
is made by completing this run. No analysis/tuning is performed during execution.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_occlusion_study.py \
  --spec configs/library_occlusion_20261122.json \
  --output results/library_occlusion_20261122 --execute
```

Omit `--execute` for a no-write/no-raycast plan. Each target worker has a
600-second timeout. Use a new root for every attempt; no automatic retries.

## Geometry Sweep Extension (2026-11-22)

`configs/library_occlusion_large_20261122.json` uses the explicit
`library_paired_occlusion_sweep_v1` schema. The earlier v1 target-list guard
remains 16 maps. The sweep is bounded to 64 maps, 600 seconds per case and
7,200 seconds for the batch, with 2 GiB free-disk checks before each case.
Candidate/target/matrix and exact-subset enumeration limits are unchanged.
Each worker runs sequentially with a fresh target scene; no parallel fan-out.

- A: widths 8/9/10/11/12/13 m crossed with junction centers (32,44), (40,60),
  (60,40), (68,56) m, yielding 24 T-junction geometries in 100 m maps.
- B: lengths 120/160/200/240 m, widths 8/12 m, turns 15/30/45/60 degrees,
  yielding 32 straight-arc-straight corridors. Bend length 32 m, mesh step
  2 m and two composition modules remain fixed.
- 56 geometries x two occlusion conditions = 112 sequential jobs;
  each has 34 arms x three K values = 11,424 method/K comparison rows.
  Each job also computes three independent full-finite-domain references.
- Libraries, sensor models, candidate/target sampling and screen recipe stay
  fixed. Query budget is the unique transferred pool size on each geometry,
  matched across methods and across that geometry's clear/occluded pair.
  The four original maps' budgets must not be imposed on these different maps.
- All axes and resolved inputs are frozen before raycasting. No geometry is
  selected based on method performance. Invalid geometry is rejected before
  execution; failed jobs stop the batch and are not silently omitted/retried.
- These are new parameter combinations in already developed synthetic families,
  not real-road holdouts, independent random map samples, or dense city-scale
  optimization. Some A configurations are geometrically symmetric. Larger B
  lengths with K fixed can be range-limited; this is not automatically a
  transfer error. No generalization or minimum sensor-count claim follows.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_occlusion_study.py \
  --spec configs/library_occlusion_large_20261122.json \
  --output results/library_occlusion_large_20261122 --execute
```

No live log reading, performance interpretation or condition tuning is part of
this execution. Final process status and integrity checks are separate from
later performance analysis.
