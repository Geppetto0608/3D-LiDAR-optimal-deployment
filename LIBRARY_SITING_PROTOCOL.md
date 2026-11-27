# Frozen Library Installation-Exclusion Study

Protocol date: 2026-11-23 KST. Frozen before execution; 224/224 cases subsequently
completed on the same day at 13:55:06 KST. Post-run analysis is in
`results/library_siting_large_20261123/analysis_20261123_v2/`, reproduced with
`library_siting_report.py --root results/library_siting_large_20261123 --output <new-directory>`.
The frozen protocol copy inside the experiment is preserved. This is a synthetic development stress test, not an operational RSU
standard, real-road validation, city-scale optimization or RL experiment.

## Question and Fixed Inputs

Does the surviving geometry-transferred library pool remain useful when some
installation sites are unavailable, with and without static occlusion?

- Reuse the A/B sources and 56 geometries in
  `configs/library_occlusion_large_20261122.json` without expanding the library.
- A: 24 T-junction geometries, 100 m square. B: 32 straight/arc/straight
  corridors, 120-240 m long, two modules. These are developed synthetic families,
  not arbitrary networks or 56 independent real-road samples.
- Freeze the source-derived candidate/target resolution, LiDAR model, K=1,2,3,
  alternatives, transfer rule, optical controls and random seeds 0-29.
- Cross two installation levels (open/restricted) with two mesh levels
  (clear/occluded). 224 worker cases, 34 arms each, 22,848 comparison rows and
  672 separately stored full-reference K rows.
- Sources live under ignored `results/`; a clone alone does not include them.

## Geometry-Only Constraint

One closed world-XY rectangle forbids every pose whose XY lies inside or on its
boundary, regardless of height/yaw/pitch. It changes neither targets nor mesh.
The test uses the same inclusive rectangle predicate as `CoverageRequirements`.

- A: east approach, center 22 m from the junction, longitudinal extent 20 m.
- B: station 0.25 times corridor length, longitudinal extent 20 m. The complete
  band must lie on the first straight approach; unsupported recipes fail.
- Transverse extent is the road plus both sidewalks, so both installation
  sides are excluded within the band. This is a hypothetical unavailable-site
  band, not a measured clearance requirement.
- Open level has no forbidden rectangles. Static screens retain the preceding
  study's 4.5 m height, 0.2 m thickness and 70% road span.
- Bounds are resolved from geometry before any query planning/raycast, never
  from target scores, library peak locations or previous outcomes. Realized
  exclusion fractions vary with the finite candidate grid. Zero-effect cases
  remain in the study.

## Optimization and ID Spaces

For unrestricted candidate domain D and forbidden XY region Z:

```text
D_Z = {p in D : XY(p) not in Z}
Q_Z = transferred_pool intersect D_Z
q_Z = |Q_Z|
F(P; G) = sum_j 1[sum_{p in P} a(p,j;G) >= 1]
max F(P;G), subject to P subset D_Z and |P| <= K
```

The library arm selects from Q_Z. Geometry-single, geometry-union,
geometry-portfolio and each random arm select exactly q_Z eligible candidates
without physical masks, then optimize their queried pool by exact subset
enumeration. All arm decisions finish before the full D_Z reference is queried.
There is no wall-clock or ray-count matching. This study adds no costs, spacing,
pole capacities, redundancy threshold, tracking objective or policy learning.

Filtering happens after the existing geometry-only projection. It does not
reproject a forbidden pose to a nearby allowed site. A layout containing a
forbidden pose becomes unavailable as a complete-layout proposal. Its surviving
individual candidates remain in Q_Z for recombination. No oracle padding.
If Q_Z is empty, every matched arm queries zero candidates and returns the
empty deployment; that is a library-availability failure, not evidence of parity
on a meaningful positive budget. The full reference still runs. A completely
empty D_Z is explicitly unsupported by this runner and stops preflight, rather
than being silently omitted or claimed mathematically infeasible.

Current `selected_ids`/`full_candidate_ids` index the eligible target archive
(arm-local `selected_ids` retain their usual pool-local meaning). Every archive
also stores `original_candidate_ids`, mapping eligible rows to unrestricted
target rows. Nested `source_context` and `original_transfer` records retain the
unrestricted ID space. Source scores/certificates are not target certificates.

Exact certification applies only to these coarse, declared finite candidate
domains. It does not certify the continuous road, all feasible mount sites,
minimum sensor count on an actual road network, or robustness to unmodeled
obstacles. Bounds/coverage thresholds on other domains cannot be imported.

## Pairing and Interpretation

Within each installation level, clear/occluded runs must have identical poses,
targets, library pool and physical query plans. Additive boxes can only remove
visibility edges. Within each occlusion level, restricted archives must equal
the allowed subset of open poses and masks, with identical mesh/target order.

q_Z may be smaller than the open q. Thus comparisons across installation levels
combine domain restriction and library pool attrition; they are not fixed-query
budget causal estimates. Record q before/after and empty-pool cases. Comparisons
between methods within one case remain count-matched. Random seeds are repeated
selections on the same map, not 30 independent maps. Source-free planners receive
only the eligible geometry arrays, sensor profile, K and scalar q, not library
candidate identities or measured visibility.

Future analysis should report availability, forbidden-candidate removal, coverage
shortfall to the eligible finite-domain reference and conditional seed spread,
stratified by track and both factors. Keep zero/negative results. No superiority,
speedup, deployment-readiness or paper novelty conclusion follows from preflight.

## Reproduction

Plan only, no result directory, raycasting, selection or training:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_occlusion_study.py \
  --spec configs/library_siting_large_20261123.json
```

Actual computation, requires a new output path:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u library_occlusion_study.py \
  --spec configs/library_siting_large_20261123.json \
  --output results/library_siting_large_20261123 --execute
```

The runner freezes source code, both libraries, resolved cases and input hashes.
Workers run sequentially with 600 s per-case and 7,200 s batch deadlines, exact
enumeration limit 100,000, existing pose/target memory guards and at least 2 GiB
free disk checked before each case. These checks are not a system-wide RAM cap.
The batch deadline kills its own process group, including the active child.
Any failure stops execution and preserves partial outputs; no automatic resume.
Old experiments and source libraries are not overwritten or reused as new jobs.

Artifacts: root `manifest.json`, `inputs/`, `source/`, `launcher.log`,
`pair_validation.json`, `installation_validation.json`; per case
`runs/<map>_<open|restricted>_<clear|occluded>/` contains `report.json`,
`comparison.csv`, `target.npz`, `mesh.npz`, frozen inputs/source and `worker.log`.
Validation uses fresh raycasting with the same physical implementation, not an
independent sensor simulator. No need to tail logs during execution.

Tests:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_installation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_occlusion_study.py
```
