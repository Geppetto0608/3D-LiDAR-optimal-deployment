# A/B Library Storage Contract

Date: 2026-11-18, Asia/Seoul. First implementation contract, not evidence of
transfer benefit, large-network scalability or trained-policy generalization.
Existing experiments, source libraries and CLI defaults remain unchanged.

## Scope And Ownership

- A: whole-topology source deployments using the existing parametric road model.
- B: straight/bent component source deployments using the existing corridor model.
- Shared: `src/core/deployment_library.py`, finite-domain coverage frontier and
  minimum-count brackets. No target-map answers enter source construction.
- Independent validation: small combinatorial oracles and fresh raycast of saved
  selections. Large solver runs and timing comparisons run separately, not during
  concurrent development.

The first builders produce source libraries; separate target adapters below
handle small transfer/composition tests. Automatic retrieval and general road
network decomposition are not implemented. B is not yet an arbitrary
intersection-module library. Learning from either library remains reserved.

## What A K Row Means

For a fixed source geometry G, eligible finite candidate domain D_G and target
set T_G, let A_i be the static range/FOV/mesh-LOS mask of candidate i.

```text
d_j(S) = sum(i in S) 1[j in A_i]
f_G(S) = sum(j in T_G) 1[d_j(S) >= 1]
C_G(S) = 100 * f_G(S) / |T_G|
f*_G(K) = max { f_G(S) : S subset D_G, |S| <= K }
```

K is an at-most budget, not a requirement to install redundant sensors. Save both
`budget_k` and `num_selected` with the actual poses. If five sensors attain the
same best coverage at K=5 and K=6, both rows may store five sensors. This does not
prove five is the minimum unless the smaller-budget upper bounds exclude it.

Geometry, sensor settings, target sampling, candidate domain and objective must
be identical across a frontier. A different geometry, installation rule,
constraint profile or target weighting is a different optimization problem.
The new first-stage contract is **unweighted union coverage only**, without
cost, spacing, pole capacity, redundancy or service thresholds. Each candidate
must individually satisfy the builder's declared installation rules. No joint
constraint guarantees should be inferred from these source layouts.

## Incumbents, Bounds And Alternatives

For each requested K, keep the highest-coverage feasible layout among previous
budget layouts, deterministic greedy prefixes, supplied verified proposals and
exhaustively enumerated subsets when the declared enumeration guard permits it.
Tie order is fewer sensors, then lexicographically smaller candidate IDs.
Previous smaller-K layouts remain eligible; stored incumbent coverage cannot
decrease with K. Poses are NOT required to be nested across K.

Let L_K be the stored incumbent count. For this pure union problem, valid bounds
available without a solver are:

```text
C_ceiling = |union(i in D_G) A_i|
b_1 >= b_2 >= ... = sorted single-candidate coverage counts
U_K = min(C_ceiling, sum(r=1..min(K,|D_G|)) b_r)
L_K <= f*_G(K) <= U_K
relative_gap = (U_K - L_K) / L_K
```

Complete subset enumeration can replace U_K by the exact optimum. Equal L_K and
U_K certifies `discrete_optimal` in the declared finite domain; otherwise the row
is `best_known`. No user-approved near-optimal tolerance is currently set. Solver
status, a time limit or a requested gap of zero alone is not a certificate.
Zero L/U gives gap zero; zero L with positive U has an undefined relative gap.

Enumeration is all-or-nothing: plan `sum(r=0..min(K_max,M)) choose(M,r)` first.
If this exceeds `enumeration_limit`, do not partially enumerate or silently prune
the domain. Use greedy/provided layouts and the analytic bounds, and report the
fallback. The guard limits subset count, not total RAM or execution time.
The new builders do not invoke MILP yet; the existing MILP source results remain
separate until a domain-checked importer/solver integration is implemented.

Keep up to `alternatives_per_k` ranked feasible layouts, including the incumbent.
Each alternative records its loss from that incumbent. Equal coverage masks at
different poses are not deduplicated. These alternatives are not all certified
near-optimal and are not guaranteed to span every branch/side allocation mode.
An optimized diversity policy is a later addition, not a result of this contract.
Original solver bounds are never silently attached to a changed candidate pool.

## Maximum Coverage And Minimum Count

`C_ceiling` may be below the number of targets because some targets are invisible
from every eligible candidate. Distinguish reaching this candidate-domain ceiling
from observing 100% of targets. The ceiling is exact with unlimited candidate
selection only in this first-stage model without joint installation constraints.

For an integer required covered-target count Q:

- A verified m-sensor layout reaching Q establishes an upper bound m on minimum
  count. It need not be an optimal layout to provide this feasible upper bound.
- An at-most-K upper coverage bound below Q excludes every budget <= K, giving
  a minimum-count lower bound K+1.
- Equal count bounds certify the minimum in the declared domain. Otherwise keep
  the interval or a missing witness, rather than claiming a minimum.
- Q above `C_ceiling` is unattainable in this declared domain, not necessarily
  on the real road or in a larger candidate space.
- Missing K rows, flat incumbents, and failure to find a layout are not by
  themselves infeasibility or minimum-count proofs.

Store brackets both for `C_ceiling` and for all targets. A user service threshold
can be queried separately; no operational coverage threshold is invented here.

## Common Stored Fields

`schema=deployment_library_frontier_v1` includes:

- Context: A/B track, source geometry, sensor, candidate/target sampling and
  coordinate convention supplied by the builder.
- Domain hash: candidate poses, target coordinates, packed incidence masks and
  context. This is not a substitute for execution source/input provenance.
- Candidate/target counts, pure-union objective, at-most-K mode and requested Ks.
- K rows: actual sensor IDs/poses/count, covered count/percentage, marginal gain
  since the previous **requested** K, upper bound, relative gap, certification,
  selection origin and ranked alternatives.
- Enumeration status/guard and frontier computation time, separately from
  geometry generation, raycasting, independent verification and file output.
- Candidate-domain coverage ceiling and minimum-count bracket summaries.

Builders additionally store masks/targets, relative coordinates, configuration,
geometry and code/input provenance. In B, entry/exit port geometry describes how
to connect modules; it does not certify coverage in an unseen neighboring module.
Source masks apply only to their original scene. New target-map occluders,
interfaces and neighboring-road interactions require fresh physical evaluation.

## Validation And Release Boundary

Required checks are increasing K, finite arrays, unique candidate poses and IDs,
selected count <= K, mask union/record agreement, monotone incumbents, valid
bounds and separate alternatives. Small independent enumeration tests check
optimal counts; physical builders check coordinate round-trips and saved layouts
against a separate raycast provider.

Default invocation is a no-write/no-raycast plan. Execution requires `--execute`
and a new output directory. Never overwrite existing results or rewrite old
absolute-path provenance. Outputs belong under ignored `results/` paths.
Source-library generation and smoke validation are not source-to-target benefit
experiments. Retrieval, transfer, mixed-module composition, independent test-map
families and matched end-to-end timing are the next stages.

## Builder Entry Points

- `library_topology.py`: existing straight/T/cross parametric geometry. The new
  opt-in default `configs/library_topology_smoke.yaml` explicitly declares coarse
  10m representative bins, a single height/orientation and one target height.
  Use `--scenario-id` or an existing explicit junction `--scenario-json`.
- `library_components.py`: one straight/bent finite corridor. Explicit station,
  side, height/yaw/pitch and target sampling; entry/exit ports and boundary context.
- Both produce `frontier.json`; A archives are `candidates.npz`, `mesh.npz`,
  `geometry.json`, `canonical_candidates.json` and `validation.json`. B archives
  are `library.npz`, `geometry.npz` and `metadata.json`.
- Both freeze source/inputs with hashes and run a separate worker. Completed
  results include independent source-layout raycast checks. A/B physical
  coordinate conventions are explicit and intentionally not forced to coincide.
- These new formats are not directly loadable by the old `kK/` prototype transfer
  loader. Dedicated target adapters are `library_topology_transfer.py` and
  `library_component_transfer.py`; automatic multi-source retrieval is still deferred.

The exact smoke commands and numerical results are in README's A/B source-storage
section. They are source-only validation on small domains, not new-map evaluation.

Common contract tests:

```bash
.venv/bin/python tests/test_deployment_library.py
.venv/bin/python tests/test_library_contract_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_topology.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_components.py
```

## First Target Evaluation

`library_transfer_study.py` connects the new formats to explicit development
targets. It plans without raycasting, solving or writing unless `--execute` is
present. Execution requires a new output root, snapshots the complete input
library and implementation, and runs a frozen worker. Original source hashes
are checked again after completion. Source coverage/bounds are never transferred
as target labels or certificates.

**A mapping:** preserve the installation branch/side and fractional station
`u=(s-s_min)/(s_max-s_min)`, rebuild `s'=s'_min+u(s'_max-s'_min)`, adjust lateral
offset for the new sidewalk centerline, and project to the declared target
XYZ/yaw/pitch grid using geometry only. Each stored incumbent/alternative is
transferred independently. A collapsed duplicate pose is unavailable, not a
zero-coverage success or an invitation to query the full reference.

**B mapping:** divide a finite target corridor into explicit equal station
intervals, assign each a copy of the source component, and map fractional
station, installation side, height and inward-normal-relative yaw/pitch. Target
sampling is generated independently before transfer; projection stays within
the assigned interval and side. Ports describe actual target coordinates and
tangents. There is no artificial wrap, and visibility crosses internal interval
boundaries in the assembled mesh. This is a corridor composition adapter, not
automatic intersection-graph decomposition or a library of diverse module types.

For either track, let `D` be the independently generated full target domain and
`R subset D` the union of successfully mapped candidate poses. Evaluate:

```text
L_R(K) = best verified union coverage using at most K candidates from R
L_D(K) <= f*_D(K) <= U_D(K)
0 <= f*_D(K) - L_R(K) <= U_D(K) - L_R(K)
```

The pool can recombine individual poses from different stored alternatives and
budgets; it need not choose an entire source deployment unchanged. B has a global
at-most-K budget, not an implicit one-sensor-per-module constraint. A reduced-pool
optimality certificate does not certify the full domain. If full enumeration
completes, its bound equals its objective; otherwise preserve best-known status.
The geometry-only full-reference candidate construction receives no target
solutions, and its optimizer receives no library warm start.

Evaluation order is fixed: raycast only `R`, select and freeze its frontier,
then build the full-reference masks in a separate provider and compute the full
frontier plus GREEDY. Finally check every stored layout against fresh target
raycasts. Source masks are used only for source file consistency checks. The
full target matrix is an evaluation artifact, not an online input. No source
certificate, threshold guarantee or minimum sensor count survives transfer
without target verification.

Complete transferred layouts are also retained as feasible pool proposals. If
enumeration is skipped, the best transferred layout is not lost merely because
pool GREEDY returns a worse union. This does not warm-start the full reference.

The first declared targets, before examining their coverage, are
`configs/library_target_a_w9.json`, `library_target_a_w12.json`,
`library_target_b_turn45.json`, and `library_target_b_turn60.json`, with K=1/2/3.
They are small synthetic development cases, not untouched final test families.
A uses the 7m T source; B uses the same 80m bent source twice in a 160m corridor.
Sampling remains deliberately coarse. Equal-query source-free controls are now
an opt-in evaluation, described below. Repeated cold-start timing, broader source
retrieval, full-size source solver support,
additional installation constraints and learning remain follow-up work.

Outputs are `report.json`, `comparison.csv`, `target.npz`, `mesh.npz`, manifest,
frozen source/inputs and a worker log. Separate counters cover online pool,
full reference and fresh verification; timing phases share a process/mesh and
are diagnostics, not a controlled speedup experiment. Source build time remains
an additional offline cost. The README records exact commands and measured
results; no run is launched by this document.

## Matched-Query Controls

`library_transfer_study.py --control-seeds 0 1 ... 29` activates the separate
matched-query evaluator; expand the seeds rather than passing a literal ellipsis.
Omitting the option retains the original evaluation path. Fixed protocol:
[LIBRARY_QUERY_CONTROLS.md](LIBRARY_QUERY_CONTROLS.md). Every source-free planner
receives only full target poses, target coordinates, the optical profile, K and
q=library pool size. It receives no source assets or measured target masks.

All plans are frozen before physical evaluation, all deployments before the full
reference. Each arm gets a separate provider, exactly q distinct pose queries,
and the same exact at-most-K subset selection. No approximate fallback is allowed
in this quality control study. Source geometry/mapping costs and source creation
remain additional work; equal queries do not establish equal time or rays.

The 2026-11-18 development run completed four maps and 136 method/seed combinations.
Library matched the finite-grid optimum in 10/12 K conditions, but two geometry
controls also matched 10/12. B had no library quality advantage over any geometry
control. README records all outcomes, including A losses; this is not a source
reuse superiority or generalization certificate. `library_query_report.py`
checks completed saved artifacts and prior-result reproduction without solving
or raycasting and writes a new derived report directory only.
