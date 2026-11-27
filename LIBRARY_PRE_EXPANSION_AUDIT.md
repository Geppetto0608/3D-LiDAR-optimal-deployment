## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate, local code and numerical audit
- Origin Date: 2026-11-22 (KST)
- Verification Status: ANALYZED
- Version Label: library_pre_expansion_audit_v1
- Source: current A/B library code and `results/library_query_controls_repeat_20261119/`
- Scope: regression tests, independent selection oracle on saved masks, input probes,
  and small clear-scene FOV counterexamples. NOT a complete physical experiment rerun.

## Fix Follow-Up (2026-11-22)

The two P2 findings below have now been fixed at the user's request. The original
audit sections and original evidence remain historical records, not the current
unfixed status. No old result file or frozen source snapshot was modified.

- Track A now rejects unknown source/target scenario fields. This includes
  constraints, obstacles, cost profiles and misspelled keys, even with empty/null
  values. Existing documented provenance/OSM metadata is allowed. Nested geometry
  continues to use its existing strict schema. The legacy scenario loader outside
  this opt-in adapter was not changed. This is rejection, not constraint support.
- `src.environment.angular_visibility_mask` implements angular FOV, maximum range
  and noncoincident targets. It is used by direct raycasting, cached coverage,
  MAPPO NumPy evaluation and Torch reward, and both optical surrogate paths.
  The hidden `forward > 0.5` slab was removed. Actual mesh occlusion remains.
- Limited-FOV configurations carry `optical_model=angular_fov_range_v2` in their
  hashes and new A/B source metadata. Old/unversioned limited-FOV libraries are
  rejected on load and require rebuilding. Default 360-degree hashes/contracts
  remain compatible; the LOS cache model itself was not changed. Coincident
  points have undefined direction and are consistently excluded in all paths.
- New tests cover rejected fields before source loading/geometry creation,
  versioned limited-FOV source roundtrips and old-library rejection, 0.25/0.5 m
  targets, rear-sector visibility at 270 degrees, rotation, real wall occlusion,
  cached/direct/control parity, and MAPPO NumPy/Torch reward. Tests ran on CPU;
  CUDA numerical execution was not checked.

Verification: 195 tests ran, **194 passed and one skipped**. The skipped test is
the existing opt-in `--roadside-parity` full candidate parity check in
`tests/test_common_placement.py`, not a failed test. Python syntax checks and
three legacy CLI plus four A/B target dry-runs passed. The dry-runs created no
experiment directories.

Additionally, all **151 archived target candidates** (30+25+48+48) were freshly
raycast against their archived meshes using the current 360-degree optics.
Every mask matched exactly. Independent bitset enumeration still matched all
420 saved K rows. Both unsupported-input probes are now rejected, and the 90,
270 and 360-degree analytic probes have no mismatch. This checks consistency with
the existing geometry, not real-world sensor accuracy or occlusion robustness.

Evidence (ignored generated paths):
[regression command](results/library_audit_fixes_20261122/regression.json),
[regression log](results/library_audit_fixes_20261122/regression.log),
[syntax and dry-run commands](results/library_audit_fixes_20261122/preflight.json),
[fresh physical and selection recheck](results/library_audit_fixes_20261122/physical_recheck.json).
The earlier `diagnostics.json` in that root predates the optional archive-recast
flag; `physical_recheck.json` is the final verification run.

Reproduce the final recheck with a new output filename on repeat:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tests/audit_library_pre_expansion.py \
  --root results/library_query_controls_repeat_20261119 \
  --source-a results/library_ab_smoke_20261118/a_v2 --recast-archives \
  --output results/library_audit_fixes_20261122/physical_recheck.json

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B -m unittest \
  test_pipeline test_deployment_library test_optical_visibility test_common_placement \
  test_local_mappo test_connected_hierarchy test_geometry_controls test_installation_domain \
  test_shape_pilot test_connected_constraints test_conditional_batch \
  test_library_component_transfer test_library_components test_library_contract_audit \
  test_library_control_evaluation test_library_expansion test_library_query_controls \
  test_library_query_report test_library_topology test_library_topology_transfer \
  test_library_transfer_evaluation
```

Remaining study limitations are unchanged: the two B targets have no effective
occlusion, and top-Q alternative storage does not ensure diversity. No new
obstacle/constraint benchmark or learning run was started by this fix.

## Original Audit (Before Fixes)

## Verdict

The current finite-domain union optimizer agrees with independent enumeration.
The audit found two implementation issues to address before extending the input
or sensor model, and an important limitation of the current B benchmark.
No production algorithm, sensor default, frozen source library, or previous
experiment result was changed. No new constrained study or training was started.

## Findings

### P2: Track A silently accepts unsupported constraint and obstacle fields

Location: `library_topology_transfer.py:212`, `prototype_offline.py:63`, and
`src/environment.py:295`. The explicit scenario loader accepts extra top-level
fields, while the target builder records `operational_constraints=[]`.

Reproduced by adding each of the following separately to the existing A/w12 input:

```json
{"constraints": {"max_sensors": 0, "forbidden_xy_boxes": [[0, 0, 100, 100]]}}
{"obstacles": [{"min": [0, 0, 0], "max": [100, 100, 20]}]}
```

Both inputs were accepted. Candidate poses, target coordinates, transferred pool,
mesh vertices and triangles remained exactly unchanged. These are deliberately
unsupported fields, not a proposed supported schema. Their silent acceptance can
make a future configuration appear to apply restrictions that never enter the
calculation. Track B's target loader already rejects unknown fields.

Impact: the existing no-constraint A/B results are not invalidated. However,
adding JSON fields alone does not implement a constrained experiment. Reject
unsupported physical/constraint inputs at the A/B adapter boundary first. When
implemented, restrictions must affect eligibility/joint feasibility, visibility
where appropriate, and provenance hashes, not merely report metadata.

### P2: An undeclared forward-distance gate changes limited-FOV visibility

Location: `src/core/visibility.py:56`, also present in the cached evaluator and
`src/methods/library_query_controls.py`.

For horizontal FOV below approximately 360 degrees, the mask additionally requires
`forward > 0.5`. The returned `forward` is a sensor-frame coordinate in metres,
not a unit direction cosine or Euclidean minimum range. This extra slab exclusion
does not follow from the documented angular-FOV/range model.

Using yaw=0, pitch=0 and an obstacle-free ray path:

| Horizontal FOV | Target relative to sensor | Angular/range model | Actual mask |
|---|---|---|---|
| 90 degrees | (0.25, 0, 0) m | visible | not visible |
| 270 degrees | (-1, sqrt(3), 0) m, azimuth 120 degrees | visible | not visible |
| 360 degrees | both points above | visible | visible |

Impact: the current A/B sensor has 360-degree horizontal FOV, so this gate is
inactive there. Before introducing other profiles, either define the additional
exclusion explicitly as part of the physical model or remove/version it
consistently across direct, cached and learning/control paths. Existing parity
tests preserve the same gate; agreement between these paths is not independent
validation of the optical formula. Do not silently alter historical physics.

### Benchmark limitation: B currently has no effective occlusion

The archived full-domain physical masks were compared with the same production
optical model without raycast occlusion. An edge means a candidate-target pair,
not a unique road point or a submitted ray.

| Target | Candidates | Optical-visible edges | Edges removed by occlusion | Affected candidates |
|---|---:|---:|---:|---:|
| a_w9 | 30 | 55,507 | 2,782 | 6 |
| a_w12 | 25 | 60,122 | 3,477 | 10 |
| b_turn45 | 48 | 5,050 | 0 | 0 |
| b_turn60 | 48 | 5,092 | 0 | 0 |

Thus the geometry-only optical surrogate equals physical visibility for EVERY
candidate in these two B cases. B still tests geometric deformation, transfer,
cross-module coverage and global K selection, but it does not test adaptation to
occlusion. Its current ties are not evidence that obstacles or constraints would
be handled equally well. A does contain effective building occlusion. This audit
compares archived arrays; it does not independently raycast all four domains.

### Storage limitations, not coverage bugs

- `src/core/deployment_library.py:133` retains alternatives by coverage, then
  sensor count and candidate IDs. Top-Q storage does not enforce geometric,
  branch/side or coverage-response diversity. Some alternative poses can be
  close together. More alternatives do not automatically add new useful modes.
- `alternatives[0]` includes the incumbent by contract. A/B adapters process the
  incumbent and alternatives separately, so transfer record counts include this
  repeated layout. Candidate pools are set-deduplicated; coverage and query
  budgets are not doubled. Record counts must not be called distinct layouts.
- A single requested K and a joint K sweep need not produce the same transfer
  pool. A imports requested source budgets; B imports budgets up to max(K).
  The frozen comparison uses K=[1,2,3] throughout. Keep this policy explicit in
  future per-K or library-size ablations.

## Calculation Checks

The implemented objective is unweighted discrete target union, not detection
accuracy, continuous road area, or a constrained service objective:

```text
F(S) = sum_j 1[there exists i in S with A_ij = 1]
F*(K) = max_{S subset D, |S| <= K} F(S)
gain(i | S) = sum_j A_ij * 1[no candidate in S covers j]
L_K = coverage of the best retained feasible layout
U_K = min(coverage of the union of all candidates,
          sum of the K largest single-candidate coverage counts)
L_K <= F*(K) <= U_K
```

The analytic U applies when enumeration is not completed. Complete enumeration
sets U=L=F*(K). The sum of single counts is an upper bound even though it counts
overlap repeatedly. At-most-K permits fewer than K sensors and monotone coverage;
layouts at successive K need not be nested.

For a required count T, U_K<T excludes every budget up to K. A feasible m-sensor
layout attaining T gives an upper bound m on the necessary count. Equal incumbent
coverage at successive K alone does not certify a minimum count. These rules are
consistent with the implementation and existing brute-force contract tests.

The target evaluator uses the same full-domain IDs and target order for library
and controls. Query plans precede physical queries, and the full-domain reference
is computed after arm decisions. Source optimality is not transferred to targets.
Equal pose-query counts are not equal ray counts, total compute or wall time.

## Verification Results

- Existing pipeline: 7/7 tests passed.
- Core deployment library: 7/7 tests passed.
- `test_library*.py`: 117/117 tests passed. Total: 131, no skipped tests reported.
- Python syntax checks passed for the audited pipeline/library files.
- Separate Python integer-bitset enumeration, without calling `build_frontier`,
  matched all 420 saved K rows: 408 method/seed rows plus 12 full-domain references.
  Saved selected IDs, poses, cardinality, coverage and exact upper bounds matched.
- The two input probes and limited-FOV counterexamples are findings, not passing
  regression tests. A completed audit does not mean all issues are resolved.

The 420 checks reuse archived visibility masks. They establish selection/count
consistency on those masks, not correctness of every physical ray or real-world
optimality. The old study's fresh-scene validation also used the same physical
implementation. No new runtime advantage or generalization claim is made.

The initial test launcher incorrectly resolved the virtualenv Python symlink to
system Python and failed on missing Torch before running tests. That failed
attempt is preserved in the audit root. The successful runs under `regression/`
use the virtualenv interpreter. This was an audit launcher error, not a project
test failure.

## Reproduce

Run from `/home/robotserver/hj_ws_main` with the local source and result archives.
The diagnostic refuses an existing output file; use a new filename for repeats.
It only reads old experiments and writes its own derived report.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tests/audit_library_pre_expansion.py \
  --root results/library_query_controls_repeat_20261119 \
  --source-a results/library_ab_smoke_20261118/a_v2 \
  --output results/library_pre_expansion_audit_20261122/diagnostics.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B tests/test_pipeline.py
.venv/bin/python -B tests/test_deployment_library.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B -m unittest discover -s tests -p 'test_library*.py'
```

Evidence: [diagnostics.json](results/library_pre_expansion_audit_20261122/diagnostics.json),
[regression commands and exit codes](results/library_pre_expansion_audit_20261122/regression/checks.json).
The diagnostic includes input and audit-code SHA-256 hashes. Generated evidence
is in ignored `results/`; a repository clone alone does not include it.

## Interpretation Scan

ARS fallacy checklist: 11/11 considered; this is a deterministic code audit,
not a significance test or a clinical/causal study.

| Check | Scope conclusion |
|---|---|
| Simpson's paradox | Preserve map/K results; pooled means do not imply every map improves. |
| Ecological fallacy | Repeated budgets/seeds are not additional independent road maps. |
| Berkson's paradox | Four selected development maps do not represent real-road frequencies. |
| Collider bias | No adjusted regression or causal conditioning is performed here. |
| Base-rate neglect | No detection sensitivity/specificity claim; coverage is target count. |
| Regression to the mean | No claim of new improvement from repeating the same study. |
| Survivorship bias | All four cases and all archived arms are included, including weak A/w12 results. |
| Look-elsewhere effect | No p-value/significance claim; no winner-only subset is audited. |
| Forking paths | Frozen comparison retained; selecting new hard cases makes them development cases. |
| Correlation/causation | No inference from query reduction to runtime reduction or real-road effectiveness. |
| Reverse causality | Not applicable to deterministic replay; no observational directional claim. |

## Next Gate

Before extending the library, reject unsupported inputs and add a regression
test for that rejection. Keep the 360-degree profile fixed until the separate
limited-FOV model issue is resolved/versioned. Then introduce ONE declared
installation exclusion or effective occluder, verify that it actually changes
candidate feasibility or visibility, and compare using the existing frozen
library and controls. Only after diagnosing failures should source geometry or
layout diversity be increased. No such new experiment has been executed here.
