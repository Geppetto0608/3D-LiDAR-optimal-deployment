# Geometry and Connection-Aware Deployment Reuse

## Main Research Question

Can a library of representative road deployments be transferred and composed
under geometry, connection and installation changes, then selectively refined,
to obtain feasible high-quality network placements with less target-map work
than cold-start optimization?

Detection/tracking are deferred. This is static geometric coverage, with shared
installation requirements and an explicitly chosen objective J. Neither adding
constraints nor using MAPPO establishes a contribution or an optimality guarantee.

The intended hierarchy is:

1. Offline: multiple source layouts per K, branch allocation and boundary response.
2. Transfer: road-relative coordinates conditioned on geometry and connection ports.
3. Composition: global candidate/layout choice and sensor-budget allocation.
4. Correction: selective visibility queries and local LS/MAPPO updates.
5. Validation: actual assembled geometry, global target identity and constraints.

The existing 410xxx/420xxx experiments address pose reuse on width/offset variants.
The new pilot below implements a small part of steps 1-3 and LS correction.
It does not implement the complete hierarchy, automatic road decomposition,
learned coverage response transfer, a general master optimizer, or policy transfer.

## Pair-Aware Mechanism Check (2026-11-09)

Gate 1 of the T-ITS plan is complete. See the predeclared methods and full
commands in [GEOMETRY_MECHANISM_STUDY.md](GEOMETRY_MECHANISM_STUDY.md) and the
[468-worker report](results/geometry_mechanism_20261109/REPORT.md).
The original shapes/constraints/K=2 and query budgets 1/2/4 are unchanged;
three fresh-process timing repetitions are not additional quality samples.

The added static-pair ranking scores both masks optically. The no-feedback
control shares the adaptive schedule but substitutes optical partner masks,
freezing its full plan before actual queries. Both use measured masks for the
final pair choice and independent direct validation of the selected deployment.

Adaptive and no-feedback return identical final IDs/J at every budget in all
12 cases, despite different query pools in eight q2 and ten q4 cases. Static-pair
beats adaptive in four q2 cases and ties eight. All three match the full reference
at q4, but their mean per-case median cold-ready times exceed full. This does not
support a measured-feedback benefit or full-relative time reduction here.
Static-pair is not uniformly superior: it loses to static-single in two q1 cases.

75 regression tests and the full audit passed. Earlier artifacts are preserved.
The earlier adaptive-versus-single/random advantage below remains a historical
fact, not evidence that feedback itself is needed. Keep these stronger controls;
next test useful partial-visibility bounds on tiny instances before network-scale
expansion. Neither this negative control nor a renamed pair ranking establishes
novelty, and the proposed large-network/library research is not yet implemented.

## Equal-Query Selection Validation (2026-11-04)

Freeze before execution: existing six development shapes, base/combined profiles,
K=2, budgets 1/2/4 actual pose queries per eligible site. Compare adaptive_batch,
static unoccluded single-sensor ranking, and genuinely nonadaptive random pools
(seeds 0..4, uniform without replacement within each eligible site). One fresh
process per arm, plus one unrestricted full reference per condition: 264 workers.
Random seeds are stochastic quality repetitions, not additional maps or timing
repetitions. No source library, MAPPO, detection, new thresholds or tuning.

The earlier random-prior ablation remained conditional/adaptive; it is not a
random-pool baseline. The new random selector does not compute optical scores or
use observed masks to choose subsequent candidates. A seed defines nested pools
across budgets. Every arm uses the same measured-pool exact K=2 upper selector.
The full reference is evaluation-only and is never supplied to a limited worker.

Primary checks: identical actual pose-query counts per case/budget, feasibility,
paired constrained objective J differences, full-reference gaps, seed mean/SD,
and nested-pool monotonicity. Infeasible J is null, never zero. Report all seeds,
not best-seed selection; average within case before averaging across cases.
Equal pose-query counts are NOT equal ray counts, CPU time or FLOPs: retain all
three counters and the measured latency. One timing observation per arm does not
establish a robust speedup. Known development shapes are not unseen validation;
the two profiles and mirrored shapes are not independent real-road samples.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py --methods adaptive_batch static random full \
  --query-budgets 1 2 4 --random-seeds 0 1 2 3 4 --repeats 1 \
  --output results/geometry_budget_20261104 --execute
.venv/bin/python results/geometry_budget_20261104/source/geometry_budget_report.py \
  results/geometry_budget_20261104
```

Use a new output root when rerunning. The report runs after the frozen independent
array/physical/pool-optimum audit. Completed: 264/264, all feasible. Mean J losses
at budgets 1/2/4 are adaptive 7.075/1.183333/0, static 7.075/6.675/4.341667,
random 32.4775/15.74/8.384167. Adaptive vs static is 12 ties at budget 1 and
8 wins/4 ties at budgets 2 and 4. At budget 4, reference matches are 12/12,
4/12 and 7/60, respectively (random: 12 cases x 5 seeds). Adaptive exceeds the
random-seed mean in all cases/budgets, but individual random seeds win twice at
budget 1 and once at budget 2. All comparisons retain these outcomes.

Budget-4 mean cold-ready time is 3.1244 seconds for adaptive versus 2.8842 for
full: no full-relative latency advantage. Full uses 348 mean queries; budget-4
arms each use 116. Their mean rays differ: adaptive 153921.25, static 164903,
random 133645.0333. This is a pose-query efficiency comparison, not equal rays.

Verified: 69 regression tests, syntax/CLI plans, 264 independent finite-pool
optima, 252 policy replays, 6912000 fresh visibility entries, 816 independent
sampled triangle checks, and 168 nested-pool/quality-monotonicity checks.
Separate report arithmetic checks cover 252 limited rows and 36 seed SDs.
Previous geometry studies retain 2669 artifact hashes and 84 paired quality/
deployment matches. See results/geometry_budget_20261104/REPORT.md (ignored).

Interpretation is limited to advantage over these two simple selectors. Pair-aware
scoring and observation feedback both differ from static single-sensor ranking;
the experiment does not isolate feedback's causal contribution. A pair-aware
nonadaptive control and unused geometry/occlusion tests remain necessary before
stronger mechanism, generalization, novelty or large-network claims.

## Behavior-Preserving Batch Screening (2026-11-03)

`geometry_placement.py --methods adaptive_batch` is an opt-in CPU optimization.
The scalar `adaptive` default and historical experiment settings are unchanged.
Each round caches measured partner masks, packs predicted/known intersections
in blocks of 16 candidates, and screens the same necessary global/interface
constraints before scoring. Where multiple partners survive, the complete
scalar objective dot product, rounding and ID ties are retained. Single
survivors still require the shared full assessment. Custom requirement
subclasses or disabled screening fall back to scalar. No extra true visibility
is queried and no current-round proposals enter the frozen partner context.

The fresh-process comparison is the same 12 known development conditions at
budget 4: scalar, batch and full, each with three repetitions and balanced
cyclic launch positions (108 workers). All 108 deployments passed; scalar/batch
queries, decisions, scores, work and final deployments agree in all 36 paired
repetitions. All three arms match the eligible finite K=2 optimum in all cases.
Mean case-median selection seconds are 1.273 / 0.883 / 0.242; cold-ready seconds
are 3.548 / 3.149 / 2.893, respectively. Selection time fell 30.6% (ratio of
stage means). Batch is faster than scalar in 12/12 cases, but slower than full
in 12/12. Mean paired cold-ready ratios are batch/scalar 0.895 and batch/full
1.087, distinct from ratios of table means. Queries remain 116 / 116 / 348.

Evidence: `results/geometry_batch_20261103/` (ignored), including frozen inputs,
30 source files, 143 regression tests, 36 independent pool optima, 24 policy
replays, 108 deployment/clock checks, 6,912,000 recast entries and 816 sampled
independent LOS checks. Seventy-two matching scalar/full runs also agree with
the previous geometry study; recorded prior/response artifacts are unchanged.
This establishes an implementation improvement, not new algorithmic novelty,
network scaling or latency superiority over full. Timing/server limitations
below still apply; previous measurements are preserved rather than overwritten.

## Source-Free Default and Timing Protocol (2026-11-03)

Operational default: `geometry_placement.py`, adaptive geometry with four queries
per eligible station/side and one fresh worker per case. It requires no source
assets and does not create a full target cache/reference by default. Source
reuse remains a separate information-value experiment. Legacy CLIs and frozen
experiments are unchanged. The new path remains two modules and fixed K=2.

The explicit comparison uses six previously observed target shapes, base and
combined profiles, adaptive/static geometry at budgets 2/4 and an eligible full
reference, repeated three times: 12 cases, five arms, 180 fresh CPU processes.
The static control ranks unoccluded single-sensor J per site; adaptive scoring
also conditions each unmeasured optical mask on measured partner masks. All
methods finally select from real queried masks using the same constraints and
finite-pair objective. No source model, LS, MAPPO or Gurobi is invoked.

Process launch order is deterministically rotated by case and repetition. Full
reference runs may be interleaved for timing-order balance, unlike prior studies'
online-before-reference scheduling. No reference arrays/results are supplied to
any adaptive/static worker: its only data inputs are frozen geometry/requirements
and physical config. Each worker constructs its own scene and query cache.
This is code/data-flow isolation, not an OS filesystem security sandbox.

Cold algorithm-ready time is measured from the parent launch checkpoint to the
child algorithm-complete checkpoint on the same Linux CLOCK_MONOTONIC clock.
It includes interpreter/module imports, input/config loading, geometry/scene,
predictions, physical queries and exact upper selection. It excludes fresh final
validation, diagnostic array export, parent aggregation and the final study audit.
Child decision time and whole worker wall time are separate. Clock ordering and
component boundaries are checked; repeated quality/work/selection must agree.

Each case/arm has a median, Q25/Q75 and raw three-run records. Repeats do not
increase the map sample size, select a best deployment or establish statistical
significance. Equal-J latency ratios are reported only where both arms match the
declared full-reference objective. OS caches and other server activity are not
controlled; thread environment variables do not enforce every library thread
or CPU affinity. This is fresh-process measurement, not hardware cold-cache or
deployment acceptance.

Posthoc audit independently checks unique finite-pair optima, replays each unique
policy, checks every repeated deployment/timing, recasts only previously queried
poses and applies independent road-frame/FOV and sampled triangle checks. Without
the full arm, no global reference is fabricated and global gaps remain null.
The full arm is a finite K=2 reference, not a continuous/network certificate.
Results: `results/geometry_placement_20261103/` (git ignored). Completed: 180/180
workers, 180 feasible deployments, 135 regression tests. Mean case-median cold
ready seconds / full-J matches are: adaptive q2 2.712/6 of 12; adaptive q4
3.550/12 of 12; static q2 2.368/2 of 12; static q4 2.409/4 of 12; full
2.903/12 of 12. Query means are 58/116 for budget 2/4 and 348 for full.
Adaptive q4 uses one third of the queries, yet is slower than full in all
12 paired equal-J case medians. Static is faster but loses reference quality
in most cases. This is evidence of query reduction, not a latency advantage.

The separate selection-stage means of case medians are 1.276s for adaptive q4
and 0.238s for full query generation; exact upper selection is 0.040s and
0.419s respectively. Geometry/backend setup is about 2.1s for both. These
stage medians do not sum exactly to the median total. A query-reduced method
can therefore spend more CPU time on its conditional candidate screening.
Default q4 remains an explicit physical query budget; no hidden full fallback
is added. Without a query restriction, full was the practical faster option
for these small problems. Large-network scaling and expensive scenes are not
tested by this result.

Audit passed 60 unique case/arm finite-pair optima (including 12 full cases),
48 unique policy replays, 180 deployment/timing checks, 6,912,000 fresh
visibility entries and 816 independent LOS samples. The default-only smoke
also verified that no full reference or unvisited-pose cache is generated.
Earlier prior files (2,273) and response-transfer online artifact hashes (1,440)
remain intact. Original performance values are not overwritten or mixed with
these fresh-process timing boundaries.

## Occlusion Response Transfer Pilot (2026-10-23)

`connected_response_transfer_study.py` adds an opt-in surrogate mask transform;
legacy selection defaults are unchanged. The previously verified isolated
straight source has zero optically valid occluded entries. If `V_s = O_s`,
its residual `B_s = O_s & ~V_s` is identically zero: transferring it cannot add
occlusion information, regardless of how densely its marginal landscape is stored.
This is an information limitation of that source, not evidence against every library.

The new bounded pilot builds five separate full-length 80m sources: a straight
10m road, left/right 70-degree 10m roads with a 22m transition, and left/right
85-degree 12m roads with a 26m transition. Six target shapes are disjoint in
width/turn/bend specification: 9m/+-40/18m, 11m/+-65/22m, 13m/+-80/27m.
Four previously defined profiles produce 24 conditions; these remain synthetic
development evidence, not a real-road or independent final evaluation set.

Canonical pose mapping preserves station, installation side, height and inward
relative yaw/pitch. Target mapping preserves station/height and selects the
nearest normalized lateral source sample of the same side within half a source
sample spacing. Unsupported target samples retain target optics. Transfer only
uses positive source occlusion labels, never optical source zeros as invisibility.

`Q_t(i,j) = O_t(i,j) AND NOT B_s(g(i),h(j))`, where
`B_s = O_s AND NOT V_s`. The target optical surrogate bounds the prediction;
`Q_t` is not measured target visibility or a feasibility certificate.
Overlap/union against a measured partner uses the existing conditional rule,
so a source residual can change the marginal response, not only a unary prior.

Four controls share target optical single-J seeds, tie-breaking, fallback,
conditional scoring, fast necessary checks, query budgets 1/2/4, and exact K=2
upper selection. Only the mask transform differs: identity geometry, measured
straight-source residual, nearest-source optical-only zero residual, and measured
nearest-source residual. The optical-only control never reads measured source
masks. Source choice minimizes squared descriptor differences scaled by 7m,
90 degrees and 20m, with source-name tie-breaking; no target scores enter it.
The per-site budget-1 seed sequence must be identical across controls, and
larger budgets must be nested prefixes. Zero-residual optical controls must
reproduce geometry's policy exactly. More measured candidates preserve the
best feasible pool objective, but incorrect residuals can worsen candidate
choice at a fixed budget. No transfer error or optimality guarantee is claimed.

All source arrays/code/configs are frozen before target queries. Per-case online
records are committed before eligible full-reference raycasting. Target truth
is used only for posthoc prediction-error evaluation and independent finite-grid
pair enumeration. Frozen audit also replays policies, validates budgets and
counts, recasts complete source/unique target geometries, and checks separate
road-frame/FOV formulas plus sampled segment-triangle LOS. Recast engine and
model/policy replay are shared implementations; LOS is not independently
exhaustively proven. Feasibility failures retain explicit denominators, never
become zero scores. Source loading/mapping is inside each decision timer;
offline source construction, process startup and final validation are separate.
Ordered single executions are not a repeated cold-start runtime benchmark.

This is a deterministic mask-transfer experiment, not pretrained RL, MAPPO
fine-tuning, learned response transfer, periodic boundary conditions, detection,
or a general city-network master optimizer. Output is under the ignored
`results/connected_response_transfer_20261023/`.

Completed: 24 conditions, 288 online outputs and 24 full references, all feasible;
128 regression tests passed. Full-reference J matches for geometry vs nearest
residual are 3/24 vs 3/24 at budget 1, 13/24 vs 16/24 at budget 2, and 24/24 vs
21/24 at budget 4. Paired wins/ties/losses are 0/24/0, 3/21/0 and 0/21/3.
Both zero-residual controls exactly reproduce geometry deployments. Each method's
own incumbent J remains nondecreasing with budget; the relative ranking changes.
Budget-2 improvements occur on the two 65-degree geometries (three profile
conditions); budget-4 losses occur on the two 40-degree geometries (three
conditions). Profile repetitions and mirrored shapes are not independent roads.

Across eligible candidate-target entries in the 24 conditions, false-visible
predictions decrease from 116,320 to 4,446, but false-blocked predictions rise
from zero to 40,998. These are static visibility prediction errors, not object
detection false positives/negatives. The current deterministic nearest source
adds information but can transfer excessive occlusion. Geometry mismatch is an
observed association, not a proven causal attribution for each missing pose.
Budget-2 mean decision times are 0.416s geometry and 0.444s residual; no speedup
or uniform superiority is established. New audit checked 312 deployments,
24 full optima, 288 pool optima/policy replays, 13,132,800 recast visibility bits,
and 1,503 independent LOS samples. Original prior-study files (2,273) are unchanged.
Follow-up confidence/calibration must use measured online evidence or source-only
rules and an independently fixed evaluation protocol, not target oracle masks.

## Conditional Prior Ablation (2026-10-23)

Verification follow-up: `connected_prior_verification.py` checks the completed
study without changing its files. The strengthened saved-array auditor also
checks complete frozen input/source inventories, plan counts/sensor settings,
source arrays and alternative ordering, mesh reconstruction, CSV condition
identities, pair-domain certificates and reference work/timing boundaries.
Adversarial tiny studies test rewritten hashes and incorrect metadata; 122
regression tests passed after this hardening. Production selectors are unchanged.
The optional physical stage recasts every source pose and each unique target
geometry once, then checks all eligible archives, online observations and saved
deployments. It checks road frames/FOV with independent formulas and stratified
LOS samples with independent segment-triangle intersection. Shared-engine
recasting is not independent proof of every LOS bit. Full-grid verification
queries are offline evidence only, never supplied to online selection or charged
as its work. Output must be a new directory outside the original study; complete
before/after file inventories and hashes detect any changes to original files.
Evidence: `results/connected_prior_verification_20261023/`, ignored by git.
The completed check reproduced 7,084,800 visibility bits from 3,120 poses,
passed 920 independent LOS samples and checked 528 deployment records; all
2,273 original files were unchanged. Current unary source-prior advantage
remains unsupported; no original performance numbers were corrected.

Entrypoint: `connected_prior_ablation.py`. This isolates the current source
prior while keeping the same adaptive probing, measured-mask upper selection,
global fixed K=2 domain and hard constraints. No lower search or MAPPO.
It reuses the previously observed 24 development conditions, not new test roads.

The previous `geometry_single` control was non-adaptive. It did not establish
that the source prior caused the adaptive probe method's 24/24 result.
The new controls all use that same sequential conditional rule:

| Prior | Ranking information |
|---|---|
| source | Actual single-sensor J on the isolated straight source |
| geometry | Unoccluded range/FOV single-sensor J on the target |
| random | One fixed candidate permutation for each seed 0-4 |

The prior controls seeds, score ties and fallback when no predicted feasible
partner exists. Source-free controls do not read source scores in any of these
roles. All controls still know target geometry and use the same range/FOV
predictions for the conditional step; they are not geometry-free policies.
This tests current source single-J ordering, not every possible source-response
library, multiple prototypes, pretrained policy or learned geometry transfer.

Query budgets: 1, 2, 4 poses per eligible station/side. Each arm/budget has a
fresh physical cache; no query sharing across arms or budget tiers. Deterministic
policy prefixes are checked across budgets. Random uses the same predeclared
seeds across geometries/profiles/budgets and reports every seed, never the best.
There are 21 online arms per condition, 504 online records and 24 full references
in the default plan. These are not 504 independent maps or timing repetitions.

All online records are committed before the per-case offline reference cache
is created. Unlike the previous audit archive, this reference queries only
individually eligible poses: 480 without exclusions or 216 with exclusions.
Forbidden/unqueried masks are never supplied as observed zeros to an algorithm.
Full-reference optima and each queried-pool optimum are independently enumerated.

New opt-in prediction screening checks necessary interface, cost, distance,
coverage/overlap and single-failure conditions before the complete shared
evaluator. Tests compare screened and unscreened choices, including infeasible
predictions. All ablation arms use this same screen; old defaults and frozen
outputs remain unchanged. It is a computation change, not extra information.

The prior score vector, round contexts, proposal decisions, actual query order,
physical ray/call counters, final masks and deployment identities are retained.
The audit reconstructs geometry priors with an independent 360-degree tilted
plane calculation; source priors from source masks/costs; random rankings from
their recorded seeds. It verifies hashes, chronology, exact budgets, nested
prefixes, independent pair optima and report statistics. Recorded policy replay
shares code, and float32 ray-count accounting uses shared optics, not new LOS.

`comparison.csv` has raw case/prior/budget/seed records. `paired.csv` reports
common-feasible source/geometry differences and all-seed random mean/sample SD
(`ddof=1`), with explicit success counts. Missing feasible pairs are not zero
scores. Summary J gaps are conditional on success; feasibility and match counts
retain all requested trials in their denominator. Random trials on a case are
not independent road observations or evidence of statistical significance.

`decision_sec` times target scene initialization, prior construction, physical
probes and upper selection. Geometry/targets are built once and recorded as a
shared component, with `online_component_sum_sec` = geometry + decision.
Source building/loading, process startup/import, serialization and independent
final validation are excluded. The reference decision boundary includes its
fresh eligible cache and exact selection. These single-process ordered arms do
not establish repeated cold-start speedup; optical work/rays may differ at the
same pose-query budget.

```bash
.venv/bin/python connected_prior_ablation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_prior_ablation.py \
  --output results/connected_prior_ablation_20261023 \
  --query-budgets 1 2 4 --random-seeds 0 1 2 3 4 --execute
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_connected_priors.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_connected_priors.py results/connected_prior_ablation_20261023
```

The CLI refuses existing output roots. Source/input/package snapshots and all
generated evidence are saved under the ignored output root. See its manifest
for execution status and summary/paired/raw tables for outcomes. Any negative
result applies to this prior/rule/grid, not an impossibility claim for reuse.

### Completed Prior Results

`results/connected_prior_ablation_20261023/`: all 24 conditions, 504 online
records and 24 full-reference records completed, all feasible. The saved-array
audit checked 504 pool optima, 24 full optima, 504 policy replays and 528
deployment records. The regression suite passed 119 tests.

| Queries/site | Prior | Full-J matches | Mean J loss |
|---|---|---:|---:|
| 1 | source | 2/24 | 20.066667 |
| 1 | geometry | 5/24 | 4.608333 |
| 1 | random | 0/120 | 55.486250 |
| 2 | source | 13/24 | 2.558333 |
| 2 | geometry | 16/24 | 0.383333 |
| 2 | random | 42/120 | 2.399167 |
| 4 | source | 24/24 | 0 |
| 4 | geometry | 24/24 | 0 |
| 4 | random | 118/120 | 0.100000 |

J loss is full-reference J minus selected J, not coverage percentage points.
On paired conditions source vs geometry has 0 wins/8 ties/16 losses at budget
1, 0/16/8 at budget 2, and 24 ties at budget 4. Thus there is no observed
source advantage over geometry ordering in this ablation. Random comparisons
do not rescue that claim; at budget 2 the random mean loss is even slightly
smaller than source, although its exact-match rate is lower. Do not cherry-pick
only exact matches or only random seeds that favor a method.

The two random budget-4 misses are seeds 0 and 4 on the same condition:
`case_02_01`, exclusion/w8_right20_b16; both lose 6 J. They are not independent
failure geometries. All 120 random trials remain in that budget's denominator.

Mean pose queries per arm are 29/58/116 at budgets 1/2/4, versus 348 for the
eligible reference, averaging exclusion and non-exclusion conditions. Every
conditional arm also performs mean 835,200 optical candidate-target tests.
At budget 4 mean decision times are source 1.265137s, geometry 1.278358s,
random 1.267600s versus eligible-reference 0.502207s. These exclude imports,
source build/load, serialization and final validation; they are single ordered
shared-server observations, not a repeated latency benchmark. Fewer physical
queries did not imply faster selection on this small fixed-K problem.

The construction worker took 318.080717s; final audit/regression were separate.
`previous_output_validation.json` reproduces the old budget-4 source policy
on all 24 cases, 48 source/reference deployment arrays and the old reference
archive's eligible subset. Thus screening and reference eligibility filtering
did not manufacture an objective improvement. The helper also checks 504 CSV
condition identities. Previous results were not edited.

The appropriate next hypothesis is richer, connection/occlusion-conditioned
source information beyond a single unary-J ordering. Whether such information
helps requires a new controlled test; larger maps or more MAPPO steps alone
do not establish source reuse value. This result does not evaluate all possible
libraries, multiple source geometries or learned response transfer.

## Hierarchy Redesign Pilot (2026-10-22)

The validation audit found two distinct limitations: preserving source response
does not preserve target constrained value, and refining within original side/8m
anchors cannot repair an incorrect installation allocation. The opt-in entrypoint
`connected_hierarchy_study.py` separates these effects without changing older
selectors, local search, physics, defaults or saved results.

This is an executable baseline contract, not a learned response model or a new
optimality theorem. The current size remains two modules, exactly K=2, one
straight source, and the already observed six geometries/four constraint profiles.
The historical config name `connected_constraint_holdout.json` does not make
this reuse of those cases a new held-out evaluation.

### Upper and Lower Decisions

For measured target candidate pool Q and actual visibility masks A(p):

```text
upper: choose P in Q, |P|=2, satisfying actual shared constraints, to maximize J(P)
lower: improve z/yaw/pitch with installation station and side fixed
control: instead allow station changes within the original 8m anchor, same side
```

Upper selection compares one sensor per module against no module/side quota.
The free branch still respects cost, pole capacity, spacing, height, global
coverage/overlap, failure coverage, installation exclusions and interface
requirements. This pilot does not insert/remove sensors or choose network K.
Removing the quota changes the optimization domain, not the learning quality.
The radius control may cross the module interface; its upper allocation label
describes initialization, not a hard constraint on the final module counts.

The lower stage is budgeted deterministic coordinate descent, not MAPPO. It
retains the feasible upper placement and accepts only strict actual-J improvements.
Each accepted move and final station/side/module signature are saved. A search
stopped by its query cap is explicitly `query_budget_limited`, not a local-optimum
certificate. Fixed-site local convergence also does not certify the joint optimum.

### Information and Query Contract

Three selectors each measure four poses at every eligible station/side:

1. `source_response`: the previous frozen source-response rule, transferred.
2. `geometry_single`: top predicted single-sensor J from target range/FOV only.
   This is a non-transfer control, not true occlusion-aware visibility.
3. `probe_context`: start with the best source-J pose per site. In each of three
   rounds, rank unmeasured candidates by predicted feasible pair J against only
   already measured other-site partners. Predict the new candidate with range/FOV;
   use measured LOS masks for the partner. Break ties by source J then ID.
   If no partner yields predicted feasibility, fall back to source ordering.

All proposals in a round use the same prior-round observation snapshot. Actual
queries occur only after those proposals are decided. No unmeasured full-grid
target masks enter this rule. This uses target geometry and a full target-point
set; it is not geometry-free or zero-shot. Source information currently provides
seeds/ties only; a benefit over the geometry-only control is not presumed.

The source rule's candidate IDs are frozen before target queries; adaptive target
pools cannot truthfully be described that way. Their complete query chronology,
partner identities, prediction scores and prior contexts are instead recorded.

The query gate rejects ineligible, over-per-site or over-total-budget candidates
before invoking raycasting. Final pair selection uses measured masks, never
optical predictions. Final poses pass the shared evaluator and a fresh raycasting
scene. Target samples have global identity and no visibility clipping at the
10m service band; Euclidean physical rays remain unchanged.

Selection consumes 160 pose queries without exclusions and 72 with the existing
exclusion profile. Each lower branch has at most 16 additional distinct physical
pose queries; reusing a measured pose does not consume another query. Different
branches have independent new-query caches initialized from the same selected
observations. Raw/fixed-site/radius branches therefore are factorial ablations,
not independent cold-start timing repetitions.

Equal pose-query budgets do not equal equal rays, optical work or wall time.
Geometry/probe selectors additionally compute range/FOV for all eligible target
poses. Optical candidate/point-test counts and total selection time are saved;
physical rays, pose queries, lower work, final validation and source construction
remain separate. Optical prediction errors never certify actual feasibility.
`work.base_candidate_queries`/`extra_candidate_queries` count cache-seeded/new
observations: raw-stage new observations are selection queries, not lower-stage
queries. `comparison.csv.extra_queries` counts only additional lower-stage work.
Final independent validation rays are outside those optimization-query counters.

### Reproduction and Audit

```bash
# No-write/no-raycast plan
.venv/bin/python connected_hierarchy_study.py

# Actual bounded development run; existing output roots are rejected
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_hierarchy_study.py \
  --output results/connected_hierarchy_20261022 --lower-extra-queries 16 --execute

# Tests and read-only saved-array audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_connected_hierarchy.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_connected_hierarchy.py results/connected_hierarchy_20261022
```

The launcher freezes source, config, input specifications and package versions.
Each case commits/hashes its online outputs before constructing its full offline
target mask archive. Full and queried-pool pair optima are independently enumerated
without score pruning. Deployment metrics, per-site and extra-query budgets,
initialization, one-coordinate improvement traces and output CSV/summary are checked.
Selector/lower policy replay shares implementation code and is labeled as replay,
not an independent algorithm proof. The read-only auditor does not recast rays.

Outputs: root `manifest.json`, `summary.json`, `comparison.csv`, `REPORT.md`,
`independent_validation.json`; per-case geometry/reference arrays, selector query
logs and masks, method deployment/observation arrays, result JSON and comparison PNG.
Generated files are ignored by git and are not included in a clone.

### Completed Development Results

Run: `results/connected_hierarchy_20261022/`, 24/24 conditions, 480/480 feasible
deployments including references. Independent enumeration checked 48 full-grid
and 144 queried-pool optima; recorded-policy replay checked 72 selectors and 288
lower branches. The current regression set has 111 passing tests.

| Selector | Raw matches full-grid J | Mean raw J gap | Fixed-site matches J |
|---|---:|---:|---:|
| Source response | 15/24 | 1.275 | 18/24 |
| Geometry single | 12/24 | 3.429167 | 12/24 |
| Probe context | 24/24 | 0 | 24/24 |

The probe selector improves source-only raw J on 9 conditions and geometry-only
raw J on 12; all other pairs tie. This is the chosen penalized J, not maximum
coverage alone. Its mean raw coverage (99.325%) is lower than source-only
(99.478%); different overlap/cost tradeoffs explain why coverage alone must not
be used to rank these objective values.

There is no J difference between upper quota/free branches on these cases, nor
between fixed-site and 8m-radius lower branches. These observations do not prove
allocation decisions or station movement are generally unnecessary. Source-seed
benefit is also not isolated by the current controls.

There are 480 individually eligible full poses without exclusions, or 216 with
exclusions; the 160/72 queried poses are one third of these respective domains.
The reference mask archive deliberately includes all 480 poses even under
exclusions for cross-profile auditing. Its recorded cost is not an efficient
eligibility-filtered cold-start comparator. No speedup is inferred from it.

Recorded mean selection time is 0.090s source-only, 0.136s geometry-single and
10.497s probe-context. These are single shared-server observations, not repeated
timing estimates. Probe-context additionally computes mean 835,200 optical
candidate-target tests. Physical query savings therefore do not establish
runtime savings. Construction worker wall time was 324.03s, with the final
read-only audit and regression tests outside that interval.

`work_and_reproduction_validation.json` checks 504 observation work records
against shared float32 optical ray counts and reproduces 72 previous source-only
and reference deployments/assessments. This is not independent LOS recomputation.
The saved helper `verify_work_and_reproduction.py` reruns that post-check.

The numerical requirements remain illustrative. Before learned selection or a
network-scale optimizer, evaluate whether the target-query selector improves over
geometry-only and source-only controls, whether removing the allocation quota
matters, and which residual errors require changing installation sites rather than
only pose refinement. Detection, unseen-road generalization and theoretical
transfer-error guarantees remain unimplemented/unproven.

## Connection Geometry

Absolute map rotation is not a geometric change when all geometry, sensors and
directional infrastructure are rotated together. Relative headings, curvature,
width transitions, offsets and occluders can change the deployment problem.

For a centerline c(s), its normal n(s), and heading h(s):

```text
xy = c(s) + d * n(s)
yaw = h(s) + inward_normal_angle(side) + relative_yaw
```

The current canonical pose is `[s, d, z, inward-normal-relative yaw, pitch]`.
It differs from old canonical branch-heading schemas. Do not interchange files.

Self-periodic/cylindrical wrapping is a future special case for repeated identical
modules. It is not appropriate to identify unrelated boundaries automatically.
Even with a periodic design, visibility must use straight Euclidean rays through
the physical repeated scene. Copies, real sensor count, failure identity and finite
network outer ends must be accounted for separately.

Curvature-dependent LiDAR placement already exists; see the publisher abstract of
[Ma et al., Roadside LiDAR placement optimization in curved tunnels](https://www.sciencedirect.com/science/article/pii/S0886779826001872).
The proposed distinction to investigate is reusable connection-conditioned
deployment alternatives and selective correction, not curvature alone.
This statement is a research hypothesis, not a novelty clearance.

## Implemented Bounded Pilot

Entrypoint: `connected_corridor_study.py`.
Configuration: `configs/connected_corridor_pilot.json`.
Geometry: opt-in `ConnectedCorridorGeometry` in `src/environment.py`.
Selection: `src/methods/connected_pairs.py`.
Physics: existing `MeshVisibility`, `LidarProfile` and shared angle conversion.
Objective/constraints: existing `CoverageRequirements` and `objective_from_density`.

Legacy scenario IDs, YAML defaults, candidate ordering, cache hashes and existing
results remain unchanged. This is a separate coarse-grid geometry experiment.

### Geometry and Sampling

- Two 40m modules, total centerline arc length 80m, constant road width 7m.
- A 20m circular transition centered on the module interface. Straight portions
  precede and follow the arc with continuous tangent direction.
- Angles: 0, -15, +15, -30, +30, -60, +60 degrees. These are development samples.
- The 0-degree case is straight; nonzero angles are smooth turns, not sharp
  mitered corners. No abrupt corner, T/cross branch, varying width or slope yet.
- Existing 360-degree horizontal / 26.9-degree vertical / 30m range profile.
- A 5m sidewalk on each side; mounting lines at signed offset +/-6m.
- Ground/road, 0.5m median, sidewalks and opaque 15m-high roadside wall strips.
  The walls have 2m thickness and open corridor ends. They are synthetic
  continuous occluders, not surveyed buildings or the legacy block map.
- Mesh station spacing <=0.5m. Analytic arcs are approximated by mesh faces;
  the optimum refers to that mesh, not an exact smooth-surface model.
- The inner bend radius must exceed the outer building offset; folded offset
  strips and turns above 90 degrees are rejected.
- Targets: 1m longitudinal/lateral station bins, excluding the median, heights
  1/1.5/2m. Total 1,440 targets, 720 per module, each owned exactly once.
  Equal road-coordinate samples are not uniform physical area on a bend.
- Candidate stations: 4m bins, midpoint locations; heights 3/5m, inward-relative
  yaw -45/0/+45 degrees and pitch -20/-10 degrees, both sides.
- 240 source candidates, 480 assembled target candidates. This is not the old
  0.5m full grid or the legacy 11-action search domain.

### Library and Methods

First compute a new isolated straight 40m source using only its own targets and
mesh. This does not consume the prior 410xxx library or any target solution.
Save all source masks/canonical poses, its best single-pose J, and the best
orientation/height per station and side. With one alternative per site, there
are 20 stored alternatives, or 40 after transfer to the two target modules.

Source ranking uses the same J weights/cost formula, but no two-sensor target
thresholds. Its best single-source pose is certified only within the source
grid. The best source pose is always retained in the alternative pool.
Source masks inform source ranking; they are NOT warped and treated as true
target masks. Every queried target pose is actually raycast.

| Method | Information and search |
|---|---|
| `independent_transfer` | Duplicate the source single-pose winner, one real sensor per module; evaluate on the entire connected target. |
| `transfer_ls` | Fresh target cache, then canonical local search from independent transfer; no run if initial deployment is infeasible. |
| `library_joint` | Query the 40 transferred alternatives, then choose the best feasible pair jointly, one sensor per module. |
| `library_joint_ls` | Independently redo the library selection, then canonical local search from that feasible pair. |
| `reference_one_per_module` | All 480 target candidates; enumerate the 57,600 cross-module pairs. |
| `reference_global` | All 480 target candidates; enumerate all 114,960 distinct pairs, including both sensors in one module. |

All methods have exactly two distinct real sensors. This is a fixed-K
comparison, not minimum-K or general at-most-K optimization.
The one-per-module reference separates alternative-pool omission from sensor
allocation restrictions. General variable-K/network-level allocation is pending.

Local search uses one sensor and one adjacent grid coordinate per move:
station +/-4m, height +/-2m, relative yaw +/-45deg, pitch +/-10deg where available.
There is no side jump, sensor insertion/removal or learned policy. Original
station anchors impose an 8m arc-length radius; anchors are not recentered.
Moves may cross the interface. These actions are NOT the legacy XY actions or
a proof that intermediate installation paths are continuously feasible.

### Shared Objective and Requirements

Default J:

```text
J = covered_target_count
    - 0.05 * excess_observation_count
    - 0.2 * installation_cost
installation_cost = sum_i (10 + z_i)
```

Hard requirements are cost <=28 arbitrary units, minimum XY separation 3m,
one sensor per identical XY pole, height 3-5m, coverage >=60%, double-covered
target ratio <=0.8, worst single-sensor-failure coverage >=25%.
The cost cap excludes two 5m mounts (cost30), while two 3m mounts cost26.
Pole capacity is redundant with positive minimum spacing; it is not an
independently active factor in this pilot.

All thresholds are illustrative, not standards or operational requirements.
No power/data hubs or region constraints are included by default. Connection-band
metrics use road-coordinate target ownership, not a rotated axis-aligned ROI.

The exhaustive selector batches exact union objective values and calls the full
shared constraint evaluator for every pair that could improve the incumbent.
No feasible improvement can be discarded by this score bound, within the stated
1e-8 objective numerical tolerance. Tests compare against naive enumeration.
The returned value is a fixed-K discrete optimum only within this declared grid,
mesh, sampled targets and constraints, not an optimum of continuous placement.

### Metrics and Separation

- J, feasibility, violations, coverage, excess observations and installation cost.
- Worst surviving coverage after either of the two actual sensors fails.
- Interface band: stations within 10m of the join. Report union coverage and
  fraction seen by both sensors; not tracking, fusion or handover success.
- Longest sampled uncovered run along any fixed lateral-offset/height track.
  Runs cross the module interface, never wrap around the finite outer ends.
  Length is number of uncovered station bins times 1m. This is not a guarantee
  about continuous blind length between samples.
- Objective difference from the global coarse-grid optimum, only for feasible
  returned deployments. Infeasible/missing runs are never filled with zero scores.
- Unique target candidate queries, submitted rays, method and validation times.

Online methods have separate memoization and scenes. All four online outputs
are saved before constructing a separate exhaustive target-mask cache.
Independent direct raycasts in a new scene verify every saved deployment mask,
density, objective and constraints. This is a consistency check of the shared
physical implementation, not validation against real measurements.

The two reference selectors share their full mask build. Method timing includes
fresh scene creation and selection/refinement but excludes shared geometry,
source transfer, independent validation and export. Source build and angle worker
time are recorded separately. One process sequentially executes angles, with
ordinary import/OS caches and no controlled hardware isolation. Do not divide
these timings into a claimed cold-start speedup.

## Commands and Outputs

Plan only, no writes, raycasts or training:

```bash
.venv/bin/python connected_corridor_study.py
```

Bounded seven-angle run; use a new root when repeating:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_corridor_study.py \
  --output results/connected_corridor_pilot_20261020 --execute
```

The launcher snapshots source/configuration, records hashes/dependencies, then
executes the frozen code in a subprocess. Existing output directories are refused.
Worker failure stops the run and leaves a failed root manifest. No resume or
silent replacement of failures. All artifacts are under ignored `results/`.

```text
manifest.json / inputs/ / source/ / packages.txt / worker.log
library/library.npz / metadata.json
angle_00/ ... angle_06/
  geometry.json / geometry.npz / requirements.json / manifest.json
  reference_masks.npz / comparison.png
  <method>/result.json / deployment.npz (when a deployment exists)
comparison.csv / summary.json / REPORT.md
```

Test:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_connected_corridor.py
```

Tests cover signed/continuous geometry, non-folded offsets, target identity,
road-relative transfer, actual occlusion, rigid-rotation parity, naive constrained
enumeration, all 256 small binary-mask cases, local radius/feasibility, cross-join
blind runs, no-write planning, output guards, allocation limits, and a real small
frozen-source two-angle run. They do not establish large-network performance.

## Completed Development Run

`results/connected_corridor_pilot_20261020/` completed all 7 angles and 42 method
outputs on 2026-10-20, 10:01:31-10:01:44 UTC (worker timestamps).
All 42 saved deployments are feasible and passed fresh-scene mask validation.
No MAPPO or Gurobi solve was executed by this pilot.

| Angle | Independent transfer coverage % | Library joint coverage % | Global grid-optimum coverage % | Library J shortfall |
|---|---:|---:|---:|---:|
| 0 | 99.167 | 99.167 | 99.167 | 0 |
| -15, +15 | 99.167 | 99.167 | 99.167 | 0 |
| -30, +30 | 99.167 | 99.167 | 99.583 | 3.800 |
| -60, +60 | 99.167 | 99.167 | 99.167 | 0 |

Library joint selection queried 40 target poses versus 480 for the reference
build. It matched the full reference J at 5 of the 7 sampled angles. The two
30-degree cases have a coverage difference of 0.4167 percentage points and a
J difference of 3.8; these quantities are not interchangeable.

The direct single-source transfer already has 99.167% coverage in every case.
Library selection improves its J mainly by reducing redundant observations, not
by increasing coverage. The configured N-1 lower bound still holds, but reduced
redundancy is not inherently an operational improvement. In particular, all 42
interface-band coverages are 100%. This pilot has not exposed a connection-band
coverage failure or established a maximum reusable angle.

Library-initialized LS did not improve its initial J at any angle. In the two
30-degree cases, the recorded full-grid optimum uses the opposite installation
side from the selected library pair and a height/yaw candidate missing from that
pool. The fixed-side local action cannot reach those recorded poses, and the
original station radius further restricts search. This is evidence to inspect
source-mode omission and accessibility, not evidence that more PPO updates will
solve the problem or that MAPPO is needed.

The one-per-module and unrestricted pair optima have equal J at all 7 angles.
Thus this execution does not demonstrate a benefit from changing per-module
sensor allocation. It is also not an untouched generalization benchmark.

An independent, unpruned saved-mask audit evaluated every pair's objective and
hard checks, matched all 14 reference optima, and verified all 42 deployment
archives against the same global target/mask arrays. It uses a separate
implementation and does not call the optimized pair selector:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_corridor.py \
  results/connected_corridor_pilot_20261020
```

This command is read-only, prints its own source hash and the run manifest hash,
and does not re-raycast or train. Its supported constraint subset is explicit;
unknown constraint families fail rather than being silently omitted.

Regression verification: 54 tests, 53 passed and 1 optional full roadside-parity
test skipped; syntax checks and the three legacy method dry-runs passed.
The new corridor test file contains 10 tests, including tampered-optimum rejection.
The 60-degree computed top view in `angle_06/comparison.png` was visually inspected.

## Curvature and Alternative-Count Sweep (2026-10-21)

`connected_curvature_sweep.py` holds the 80m centerline length, road width,
fixed K=2, LiDAR profile, sampling, candidate grid, J and hard requirements
constant. It varies two factors from `configs/connected_curvature_sweep.json`:

- Signed turn angles -30, +30, -60, +60 degrees.
- Transition lengths 20, 15, 12m, producing 12 geometry cases.
- Source alternatives per station/side: 1, 2, 4. This yields nested transferred
  pools of 40, 80, 160 candidates inside the same 480-candidate target grid.

There are 36 conditions and 216 method runs, with one newly computed straight
source reused across all of them. Source ranking is fixed before target queries.
Source alternatives are prefixes of the same per-site source J ordering; the
target's coverage or optimum never determines which alternatives enter a pool.
More alternatives in this ablation do not introduce a new diversity criterion.

For a signed circular turn, `abs(curvature) = abs(angle_rad) / transition_length`.
The 60-degree radius changes from 19.10m at length20 to 14.32m at length15 and
11.46m at length12. All declared cases pass the non-folded offset requirement.
This changes how rapidly the road turns while preserving total centerline length.
It also changes straight-segment lengths, so it is a prescribed geometry family,
not a claim to isolate every other geometric quantity.

Each condition repeats the six pilot methods. Its online outputs are saved
before building that condition's independent reference cache. Reference builds
are intentionally repeated across pool sizes; they are not fed back to online
selection. The source archive is shared, but target memoization is per method.
This is a deterministic quality/query sensitivity study, not a runtime benchmark.
Angle, transition length and pool size are finite development samples, not
independent random roads or a continuous-angle certificate.

```bash
# Plan only, with no writes or raycasts.
.venv/bin/python connected_curvature_sweep.py

# Full bounded sweep; use a new output name when repeating.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_curvature_sweep.py \
  --output results/connected_curvature_sweep_20261021 --execute

# Recheck saved arrays, source-only ranking, all constraints and all pair optima.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_curvature.py \
  results/connected_curvature_sweep_20261021

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_connected_curvature test_connected_corridor test_pipeline
```

The launcher freezes code and all input files, records dependency versions,
runs a sequential worker, and executes a frozen independent audit before returning
success. The audit checks the factorial job grid, source-only ranking, target
mask/geometry invariance across pool sizes, unchanged transfer/reference methods,
all saved deployments, and every full-reference pair without score pruning.
It also verifies that best feasible library J cannot decrease for nested pools.
The same monotonicity is not assumed for LS endpoints with different initial poses.

Root outputs: `manifest.json`, `library/`, `inputs/`, `source/`, `packages.txt`,
`worker.log`, `comparison.csv` (method rows), `paired.csv` (matched conditions),
`summary.json`, `REPORT.md`, `coverage.png`, `objective_gap.png`, and
`independent_validation.json`. Each `runs/bXX_aXX/` contains its suite,
source selection IDs, the four angle outputs and a group report.

The five additional tests exercise factorial/no-write planning, physical/domain
guards, nested source-only pools and tie handling, fixed-factor preservation, and
a real eight-condition frozen-source run with independent optimum-tamper rejection.
Connection-band hard redundancy requirements and installation exclusions remain
the next constraint experiments after this geometry/pool ablation.

### Completed Sweep

`results/connected_curvature_sweep_20261021/` completed all 36 conditions and
216 method outputs on 2026-10-21, 07:53:59-07:55:06 UTC (worker computation).
The frozen independent audit subsequently matched all 216 saved deployments and
72 reference values. All saved deployments satisfy the configured constraints.
The 20m-transition/one-alternative subset also matches the original pilot's 24
corresponding method rows for coverage, J and interface coverage/overlap.

| Alternatives per source site | Target poses queried by joint selection | Geometry J matches / 12 | Mean J gap | Maximum J gap |
|---:|---:|---:|---:|---:|
| 1 | 40 | 6 | 1.866667 | 3.800000 |
| 2 | 80 | 6 | 1.866667 | 3.800000 |
| 4 | 160 | 6 | 1.866667 | 3.800000 |

Both independent transfer and joint-library selection retain 99.166667% global
coverage and 100% interface coverage throughout this sweep. Library-initialized
LS improves J in 0/36 conditions. The +/-60deg cases match the exhaustive J;
the +/-30deg cases retain a J shortfall of 3.8 at transition20 and 3.7 at
transition15/12, with reference coverage99.583333% versus library99.166667%.
These are fixed finite-grid results, not continuous-road optima.

The missing pose in each of the six recorded +/-30deg reference layouts has
source rank9 at the same installation site: height5m and relative yaw +/-45deg.
All tested source pools stop at rank4. This identifies why those particular
reference layouts are omitted, without claiming to enumerate every optimal
layout or prove which new diversity rule will perform best.

Visibility did change as transition lengths changed. Comparing the reference
mask arrays at length12 and length20 gives a maximum 2.090278% changed
candidate-target entries, at +/-60deg. This is not a deployment coverage loss.
The existing global coverage/N-1 floors remain loose for the saved deployments.
This run therefore tests reuse stability under the prescribed bends but does
not locate an interface-coverage failure or maximum reusable curvature.

`connected_curvature_report.py` creates source-rank and visibility-change CSVs,
matched-axis scientific plots and a Markdown interpretation from saved arrays.
The reviewed derived report is `analysis/REPORT.md`; `analysis/provenance.json`
records its code hash and input hashes separately from the frozen run source.
It does not modify raw result JSON or perform new optimization/raycasting.
The earlier `diagnostics/` directory is a preserved first derived-report pass;
use `analysis/` for the reviewed plots and data-derived narrative.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python connected_curvature_report.py \
  --root results/connected_curvature_sweep_20261021 \
  --output results/connected_curvature_sweep_20261021/analysis_repeat
```

The output directory must be new. Tests: 22 passed across the new sweep, the
original corridor and the lightweight pipeline, including the real small
eight-condition run and tampered-optimum rejection. Syntax and all three legacy
method dry-runs also passed. The reviewed coverage plot was visually checked.

## Source Modes and Connection Constraints (2026-10-21)

`connected_constraint_study.py` reuses one straight source across four signed
turns (-30,+30,-60,+60deg) at transition12m. Global geometry, K=2, LiDAR,
targets, candidate grid, objective and baseline requirements are unchanged.
The study adds two source rules with four candidates per station/side:

1. `top`: the four highest isolated-source J scores.
2. `diverse`: keep the best source pitch for each (height, relative-yaw) mode;
   seed at the best source J and greedily maximize minimum squared distance
   in per-site normalized height/yaw. Distance ties use source J then ID.

There are six height/yaw modes per site in the default grid; diverse selection
retains four. The rules use source data only, with equal source pool sizes.
The rule was designed after inspecting the prior development results. Its
evaluation here is not an untouched test of generalization or a novelty claim.

### Requirements and Frames

`CorridorRequirements` extends the shared bound evaluator in
`src/roadside/corridor_requirements.py`. Global J/cost/spacing/height/N-1 checks
retain the shared implementation. Additional region masks are explicitly
formed from road coordinates, while global distances and costs use world poses.
The wrapper uses an exact lookup of the declared discrete world candidates.
Unknown continuous poses raise an error; an inverse-coordinate API is future work.

The interface is the sampled target band with station in [30,50]m at default
length80 and half-band10. Four profiles form an overlap/exclusion factorial:

| Profile | Interface thresholds | Forbidden installation zones |
|---|---|---|
| base | None added | None added |
| overlap | coverage >=95%, fraction seen by both >=0.5 | None added |
| exclusion | None added | Both sides at [0,8], [34,46], [72,80]m; right side at [0,40]m |
| combined | Same as overlap | Same as exclusion |

Intervals include endpoints. Left means d>0 and right d<0 in the centerline
frame; exclusions follow the bend, not fixed world-axis rectangles. Zones alter
installation eligibility, not obstacle geometry or target membership.
All thresholds/zones are illustrative. Double visibility is not a detection,
fusion or handover guarantee. Global excess-observation penalties still apply
after satisfying required interface redundancy.

### Methods, Exactness and Work

Seven methods run for each of 16 conditions:
`independent_transfer`, `top_joint`, `diverse_joint`, `top_joint_ls`,
`diverse_joint_ls`, `reference_one_per_module`, `reference_global`.
Joint source-pool selection retains one sensor per module. The two exhaustive
references separate that quota from full legal allocation.

Each online method has a fresh target mask memo and scene. Individually forbidden
candidates are removed before selection queries; joint feasibility remains in
the full shared/wrapped evaluator. Both source rules query160 poses in base and
overlap, and72 in exclusion and combined. They need not submit the same number
of FOV-selected rays, consume equal wall time or share LS initial poses.
Local search retains the original station anchors and fixed sides.

After all five online outputs for a condition are committed, a separate cache
raycasts all480 physical candidates. Exact pair selection considers the eligible
candidate subset (480 or216) and screens interface-infeasible pairs using their
two actual masks. This screen is exact for the stated two-sensor regional
thresholds. Every prospective improvement still passes the full evaluator.
The independent audit enumerates all original pairs with exclusions and interface
requirements applied explicitly, so removing illegal candidates cannot hide a
feasible omitted solution. Unit tests also cover completely blocked domains.

Infeasible original transfers are saved with physical coverage and violation
labels, but no objective regret against a feasible optimum. Missing feasible
library solutions would have null metrics and explicit status. `validated_direct_raycast`
means mask consistency, not feasibility; these are separate fields.

### Execution and Completed Results

```bash
.venv/bin/python connected_constraint_study.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --output results/connected_constraint_study_20261021 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_constraints.py \
  results/connected_constraint_study_20261021

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python connected_constraint_report.py \
  --root results/connected_constraint_study_20261021 \
  --output results/connected_constraint_study_20261021/analysis_repeat

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_connected_constraints test_connected_corridor \
  test_connected_curvature test_constrained_coverage test_pipeline
```

The worker completed on 2026-10-21 at 08:58:04-08:58:38 UTC. All 112 method
deployments passed fresh-scene raycast consistency checks, of which 104 are
feasible. The eight infeasible cases are original transfers in exclusion and
combined profiles. Both joint source rules find feasible layouts in all 16 cases.

| Method | Feasible / 16 | Matches global finite-grid J |
|---|---:|---:|
| Independent transfer | 8 | 0 |
| Top-source joint | 16 | 13 |
| Diverse-source joint | 16 | 16 |
| Top-source joint + LS | 16 | 13 |
| Diverse-source joint + LS | 16 | 16 |

Diverse pools improve J in three conditions and tie in 13. Gains are 3.7 in
base +/-30deg and 4.15 in exclusion +30deg. In those cases coverage changes from
99.166667% to 99.583333%. There is no observed LS improvement after either joint
selection. Both references have equal J in every condition, so this execution
does not establish a benefit from variable per-module sensor allocation.

All overlap/combined selected deployments satisfy the 0.5 interface threshold.
For example combined +60deg retains 100% interface coverage and 51.1111% double
coverage, while avoiding every forbidden mount. Original transfer has higher
double coverage there but violates installation eligibility; it is not a valid
competing deployment. Required redundancy changes which feasible J is attainable.

The frozen independent audit matched 32 global/quota reference values and 32
source-pool optima, checked every saved deployment, verified equal legal pose
query counts, and confirmed unchanged physical masks across requirement profiles.
The 39-test suite passed, including actual small frozen runs, explicit blocked
domains, naive pair comparisons, frame-specific regional checks and tamper
rejection. Syntax and legacy method dry-runs passed. The derived overview and
excluded-mount scene plots were visually inspected.

Output root contains source/input snapshots and hashes, dependency versions,
one source archive, source pool IDs, per-case geometry/profile/masks/deployments,
raw/paired CSVs, report and independent validation JSON. `analysis/` adds plots
and a report with separate code/input provenance. All results are ignored.
Time scopes remain development measurements rather than repeated cold-start
benchmarks. No MAPPO, continuous optimization or real-road validation is performed.

## Fixed-Rule Held-Out Geometry Evaluation (2026-10-21)

Before querying these targets, fix the existing source diversity rule, its
four alternatives per station/side, the base global requirements, and the four
development profiles without changing any numeric threshold.
Input: `configs/connected_constraint_holdout.json`, executed by the existing
opt-in `connected_constraint_study.py` using schema `connected_constraint_holdout_v1`.

| Target road width | Signed connection angles | Bend arc length |
|---:|---:|---:|
| 8m | -20, +20 degrees | 16m |
| 10m | -45, +45 degrees | 18m |
| 12m | -75, +75 degrees | 24m |

These six tuples were not used to develop the source diversity rule. This is a
bounded synthetic held-out check, not a real-road test. Width, angle and bend
length vary together; this grid does not identify their separate causal effects.
All corridors remain 80m long and constant-width within each case. Width
transitions, slopes, junction branches and new obstacle patterns are not tested.

Build exactly one isolated straight 40m, width7m source, not one source for each
target width. Rank/select both source pools once, then freeze their IDs and
archive hashes before target visibility queries. Preserve source station, side,
height, inward-relative yaw and pitch. Replace the signed mounting offset with
`sign(d) * (target_width/2 + sidewalk_width/2)`, then shift the second module's
station by40m. The complete target reference uses the same physical grid before
pool selection. No source visibility mask is treated as true target coverage.

There are 6 geometries x4 profiles =24 conditions and7 methods per condition.
All five online choices are saved before each case's full reference mask build.
Keep source and target candidate counts at240 and480. Target sample counts grow
with width; compare J only within the same target and constraints. The mean of
raw J across different widths is not a common-scale quality measure.

The independent audit checks the frozen source geometry/ranking, transferred
canonical poses, complete geometry/profile grid, every saved mask/density and
all exact global/quota and source-pool objective values. A separate small test
uses equal angles at different widths to catch accidental case-key collisions.
No feasible pair means null quality metrics and an explicit infeasible status,
never a zero-coverage deployment or a successful tie.

Plan only (no raycasts or files):

```bash
.venv/bin/python connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json \
  --output results/connected_constraint_holdout_20261021
```

Execution uses a fresh output root and frozen source/input snapshots:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json \
  --output results/connected_constraint_holdout_20261021 --execute
```

No MAPPO training, detection, physical deployment or controlled speedup benchmark
is part of this evaluation. Once observed, these results cannot be reused as an
untouched final test after further rule changes.

### Completed Held-Out Results

The frozen worker completed at 10:14:02-10:15:00 UTC on 2026-10-21, taking
57.962 seconds internally, excluding launcher setup and the independent audit.
All24 conditions completed; all168 saved deployments passed fresh-scene mask
consistency. There are156 feasible deployments. The12 infeasible results are
the original independent transfers under exclusion/combined profiles.

| Method | Feasible /24 | Matches global finite-grid J |
|---|---:|---:|
| Independent transfer | 12 | 0 |
| Top-source joint | 24 | 17 |
| Diverse-source joint | 24 | 15 |
| Top-source joint + LS | 24 | 17 |
| Diverse-source joint + LS | 24 | 18 |

Diversity improves J against top-source pools in7 conditions, ties in11, and
loses in6. Thus the development result of16/16 optimum matches does not persist
on these new shapes. LS improves diverse initializations by2.4 J in three
width8/20-degree cases; it does not improve top-source initializations.
Six diverse+LS deployments still miss the optimum, by0.8 to10.8 J units.
Both global and one-per-module references have equal J in all24 conditions.

For width12/+75deg/base, top-source joint attains98.784722% coverage and
J=2834.85, while diverse joint attains99.6875% and J=2824.05. The second layout
has higher coverage but loses under the common excess-observation penalty.
These are differences in the stated constrained objective, not detection rates
or coverage-only regret. A diversity heuristic is not generally dominant.

The two unrefined source-pool methods each query160 eligible candidates for
base/overlap, or72 for exclusion/combined. The full reference mask archive
contains480 candidates. LS uses additional queries and target ray counts differ;
these counts do not establish equal wall time or a cold-start speedup.
Target counts are1920,2400,2880 for widths8,10,12 respectively. Canonical samples
remain evenly spaced, not uniform physical area on a bend. The source arrays,
source scores and both selected ID pools match the earlier development run.

The independent audit verifies48 global/quota references and48 source-pool
optima by complete pair enumeration, all168 saved deployments, source geometry,
road-relative transfer and equal unrefined query budgets. There are44 passing
tests including five new hold-out checks, source-width preservation, changed
target widths, equal-angle case keys, fully blocked domains and transfer tamper
rejection. Syntax and three legacy method dry-runs passed.

Artifacts: `results/connected_constraint_holdout_20261021/`, including raw and
paired CSVs, frozen inputs/source, source IDs, all case arrays and validations.
`analysis/` contains a derived report, full-condition plot and excluded-mount
scene with separate code/input provenance. Reproduce the read-only analysis in
a new directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python connected_constraint_report.py \
  --root results/connected_constraint_holdout_20261021 \
  --output results/connected_constraint_holdout_20261021/analysis_repeat

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_connected_holdout test_connected_constraints \
  test_connected_corridor test_connected_curvature test_constrained_coverage test_pipeline
```

## Saved-Result Failure Diagnosis (2026-10-22)

Entrypoint: `connected_failure_diagnostics.py`. Pure pair/action-graph helpers:
`src/methods/connected_diagnostics.py`. No physical cache generation, new formal
training or candidate-selection change is part of this analysis. Input results
are immutable; output is `failure_diagnostics_20261022/` below the held-out root.
The observed cases are now development data for any future rule.

For each top/diverse pool, enumerate all feasible global and quota optima and
all optimal quota ties. Minimize the number of absent target candidates over
the optimal pairs. A selected witness is not a unique necessary deployment.
Adding it is an oracle capacity diagnostic, not a fair same-budget improvement.
Count distinct shared source IDs separately from target candidates.

For each saved LS result, use its ORIGINAL labelled anchors and sides:

```text
full finite domain
  -> fixed initial installation sides, unrestricted stations
  -> original station radius, same sides
  -> joint feasible one-coordinate action component
  -> saved LS endpoint.
```

Adjacent coordinates are exactly station, height, relative yaw and pitch levels
from the original LS. Side changes are not actions. Invalid intermediate layouts
are excluded. The analysis also computes best single-agent multi-coordinate
relocation and a maximum-bottleneck path to the component optimum, distinguishing
plateaus/worsening moves from disconnected feasible states. None of these
counterfactual queries constitutes an online method.

### Completed Diagnosis

All24 cases,48 pool/refinement diagnoses and19 missing-candidate witness records
were processed from the stored masks. Original input hashes506/506 are unchanged.

| Pool | Raw misses | LS misses | Positive side loss | Positive anchor loss | Positive disconnection/search loss |
|---|---:|---:|---:|---:|---:|
| Top | 7 | 7 | 7 | 1 | 0/0 |
| Diverse | 9 | 6 | 6 | 2 | 0/0 |

The restriction losses are an ordered nested decomposition, not independent
causal effects. Side and anchor counts can overlap. All48 saved LS endpoints
already equal the complete anchor-domain optimum. Thus their feasible component
also has that value: the endpoint is a feasible lower bound and the enclosing
anchor domain provides an equal upper bound. More steps with the same action
domain cannot close the observed global gap. This does not establish a general
LS advantage over learning or show that coordination never matters elsewhere.

For diverse+LS, base width10/+/-45deg each loses0.8 J through side restriction.
Base width12/+/-75deg each loses5.65 J through side restriction and5.15 through
anchors, totaling10.8. Exclusion width10/+45 loses0.8 through sides; exclusion
width12/-75 loses3.6 through sides. Other diverse+LS cases match global J.

Top missing witnesses comprise seven records, source site rank9, height5m,
relative yaw+/-45deg, pitch-20deg. Diverse witnesses comprise12 records across
nine cases, source site rank2, height3m, relative yaw0deg, pitch-10deg. These are
records with repeats across cases, not12 distinct source candidates. The diverse
witnesses omit an alternative pitch within a retained height/yaw mode. Conditional
on the optimal partner, their target-site rank is1. This is an oracle context,
not a deployable candidate-selection input. Other equally optimal witnesses can
use different poses, so no unique-pose necessity is inferred.

The implication is narrower than "add pitch-10 everywhere": the two source
compression rules discard different useful responses. A proposed replacement
must preserve conditional composition value at a MATCHED pool/query budget and
be evaluated on another untouched set. Side-pattern selection and legitimate
relocation actions should be separated from the optimizer comparison.

### Verification and Reproduction

The original run's frozen independent unpruned auditor verifies96 fixed-side/
anchor optima,48 all-optimum omission checks and144 feasible counterfactual
witnesses. The48 component-optimum values use the equal-upper/lower-bound proof;
the independent audit does not verify reachable-state counts. New tests cover
nontrivial graph dips and disconnections separately on small explicit instances.
Eight new tests plus existing constraint/hold-out/pipeline tests total25 passing
tests. Tiny temporary raycasting/PPO regression tests are not new study results.

```bash
.venv/bin/python connected_failure_diagnostics.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python connected_failure_diagnostics.py \
  --root results/connected_constraint_holdout_20261021 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/validate_connected_diagnostics.py \
  results/connected_constraint_holdout_20261021/failure_diagnostics_20261022

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_connected_diagnostics test_connected_constraints \
  test_connected_holdout test_pipeline
```

Existing output directories are refused. Use a new `--output` for repeated
analysis. The independent checker is read-only unless `--output NEW_JSON` is
given, in which case it refuses an existing file. These are offline analysis
times, never an online speedup. Original/dependency hashes and a diagnostic
source snapshot accompany CSV/JSON/report/plot outputs.

## Source Marginal-Response Baseline (2026-10-22)

The opt-in `connected_constraint_study.py --response-pool` path adds two methods:
`response_joint` and `response_joint_ls`. The original seven-method path and
old results remain unchanged. This is the next compression baseline, not a
novelty claim or a learned response-transfer model.

For each source station/side site A, keep four of its12 candidates. Contexts C
are empty or any one source candidate at a different site (229 contexts/site
in the current grid). Source targets have uniform importance. With effective
coverage coefficient a, excess-overlap penalty lambda, and additive cost:

```text
R(p | empty) = J_source({p})
R(p | {c}) = J_source({p}) - (a + lambda) * |V_source(p) intersect V_source(c)|

regret(Q,c) = max[p in A] R(p|c) - max[p in Q] R(p|c).
```

Seed Q with maximum single-sensor J, which includes installation/count costs.
Add the candidate minimizing worst-context regret of the enlarged set; ties
use mean regret, highest single J, then lowest source ID. No pitch pruning or
target-derived candidate insertion is used. This is sequential greedy selection,
not exact minimax subset selection. Contexts can violate pair constraints:
the source response proxy does NOT certify downstream feasibility. Final target
optimization applies the same full shared hard constraints to all pools.

Source pool IDs and regret traces are hashed before any target raycasts. All
online methods finish before building each case's full target reference archive.
No saved target masks or oracle optimal partners are used for selection.

### Completed Development Comparison

Re-ran the24 observed width/turn/constraint conditions, not an untouched test
set. The216 saved deployments include12 infeasible original transfers; all
three joint selectors and their LS variants are feasible24/24.

| Rule | Raw optimum matches | Raw mean J gap | LS optimum matches | LS mean J gap |
|---|---:|---:|---:|---:|
| Top | 17/24 | 1.614583 | 17/24 | 1.614583 |
| Height/yaw diverse | 15/24 | 1.450000 | 18/24 | 1.150000 |
| Source response | 15/24 | 1.275000 | 18/24 | 1.150000 |

Response beats geometric diversity in three raw conditions (each+1.4 J), ties
21, loses none. It wins/ties/loses7/11/6 against top-score. After LS, response
and diversity have equal J in every condition, including the same six misses
(max10.8 J). Their actual selected IDs match in14/24, so equality is about
objective value, not a unique placement. No superiority over the stronger
post-correction baseline is established.

All three joint pools query160 legal poses in base/overlap or72 in exclusion/
combined, while the full archive has480. LS uses additional queries. Different
ray counts and unmatched latency are not speedup evidence. The independently
audited outputs verify216 deployments,48 global/quota optima,72 pool optima,
source-only selection/traces and geometry/transfer consistency. All168 original
method placements and assessments reproduced the earlier run exactly; source
arrays and top/diverse pools also matched. No MAPPO was run.

Source-response compression alone has not solved the target-context mismatch.
Next investigate connection-conditioned contexts and installation-side patterns;
separate domain enlargement from algorithm improvements and freeze rules before
any further untouched-geometry evaluation. The current small-K coarse-grid
result is not a generalization, continuous-optimality or city-scale claim.

```bash
.venv/bin/python connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json --response-pool

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json --response-pool \
  --output results/connected_response_study_20261022 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  results/connected_response_study_20261022/source/tests/validate_connected_constraints.py \
  results/connected_response_study_20261022

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_source_response test_connected_constraints \
  test_connected_holdout test_connected_diagnostics test_connected_corridor \
  test_connected_curvature test_pipeline
```

Use a new output root for repeats. Seven new tests plus existing regressions
total47 passing tests, including small actual raycasts; these are not research
samples. Syntax and standard GREEDY/MILP/MAPPO dry-runs pass. See the ignored
result root's `FINDINGS.md`, `REPORT.md`, `comparison.csv`, `source_pools.json`,
frozen `source/`, input/package records and `independent_validation.json`.

## Extended Validation Audit (2026-10-22)

Before another method or training run, the latest216 outputs were audited with
`connected_validation_audit.py`. No original experimental inputs/results or
optimizer behavior changed. Original650 file hashes remained unchanged.

### Correctness and Verification Gaps

The previous saved-array checker compared objective/feasibility but did not
fully enforce joint-pool membership, the link from a joint result to its LS
initialization, detailed assessment metrics, or coverage/gap/query/timing values
in CSV/paired/summary reports. The checker now rejects these mismatches, even
when the final objective itself matches. Added adversarial tests mutate temporary
files, never original research results.

Actual data pass the strengthened checks:216 saved deployments,48 global/quota
optima,72 pool optima, detailed component/redundancy/failure/band/module/blind-run
metrics and report consistency. All72 LS outputs also reproduce exactly from
recorded initial candidates using saved masks: final IDs, assessments, neighbor
evaluations, stopping reason and unique queried candidate counts.

Independent closed-form road-coordinate/attitude/target conversion and tilted
plane range/FOV checks pass for6 unique geometries,6,912,000 mask entries.
Independent two-sided segment-triangle intersection checks pass792 deterministic
stratified samples. This checks stored mesh geometry and sampled LOS, not all
LOS independently, actual channels, detection or real-road validity.

### What Limits the Method

Source compression: enumerate all4-of12 subsets at each site, first preserving
the original single-J seed (165 subsets/site), then without the seed (495/site).
Keep the declared minimax regret objective, mean-regret tie break and four-slot
budget. Greedy has a strictly worse worst-source-regret value at12/20 sites,
with a maximum2.65 J difference. However, exact-source representative pools
produce exactly the same raw target objective as the saved response pool in
all24 cases. Releasing the seed changes neither the selected exact pools nor
that result in this dataset.

All tied source-optimal site subsets are retained for a separate optimistic
target oracle. The two modules must share the same source library: if both
selected target sensors map to the same source site, ONE optimal site subset
must contain both source IDs. This oracle still misses9 target optima, with or
without the seed rule. Its freedom to use target answers makes it a capacity
diagnostic, not an online algorithm. The finding only concerns this particular
source response objective/context family, NOT every possible compressed library.

All72 LS endpoints maximize J within their original labelled side/anchor domain.
Top has7 positive side-loss cases and1 additional anchor-loss case; diversity
and response each have6 and2. These are ordered nested losses, not independent
causal effects. Same-domain training cannot beat those domain optima in these
cases; this does not rule out learning with different representations/actions.

The service band is NOT a coupling boundary. Cross-module visible targets reach
station offsets28.5,30.5,34.5m for the width8/10/12 geometry pairs. Many such rays
fall outside the configured+/-10m band. Current visibility/selection is NOT
clipped to that band; a future interface-only correction/reuse claim would need
a separate support-based argument. Station distance is not Euclidean range.

Remove-one-constraint analysis uses the entire finite candidate-pair domain:

| Constraint | Conditions Applied | Removing Adds Feasible Pairs | Removing Improves Optimal J |
|---|---:|---:|---:|
| Installation eligibility | 24 (nontrivial12) | 12 | 5 |
| Installation cost | 24 | 24 | 0 |
| Minimum XY spacing | 24 | 8 | 0 |
| Pole capacity | 24 | 0 | 0 |
| Height | 24 | 0 | 0 |
| Minimum global coverage | 24 | 24 | 0 |
| Maximum global overlap | 24 | 0 | 0 |
| Single-failure coverage | 24 | 12 | 0 |
| Minimum interface coverage | 12 | 12 | 0 |
| Minimum interface overlap | 12 | 12 | 11 |

Redundancy is conditional on all other enabled checks and this grid. Cost or
minimum coverage may change the feasible set without changing its maximum J;
do NOT call them generally useless. Height simply restates the current candidate
levels. The count of configured constraints is not independent difficulty.

### Scope and Reproduction

These24 conditions reuse6 observed geometries with four profiles, a single
straight source, fixed K=2 and two modules. They are not24 independent roads.
The equal canonical station/lateral samples have curvature-dependent area
Jacobian0.7000..1.3000 in the strongest bend. Sample coverage is not physical
area coverage; no historical objective has been silently reweighted.

```bash
.venv/bin/python connected_validation_audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_validation_audit.py --execute

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_connected_validation_audit test_source_response \
  test_connected_constraints test_connected_holdout test_connected_diagnostics \
  test_connected_corridor test_connected_curvature test_pipeline test_constrained_coverage \
  test_feasibility_repair test_rsu_coverage test_constrained_sensitivity
```

The audit's default root is
`results/connected_response_study_20261022/validation_audit_20261022/`.
Use `--output NEW_ROOT` for repeats. Frozen audit source and provenance accompany
`strengthened_validation.json`, `source_compression.json`, `counterfactual.csv`,
`constraint_activity.csv`, `ls_domains.csv`, `physical_checks.csv` and reports.
No new formal MAPPO training or visibility cache is built. Offline subset/pair
enumeration and LS replay are diagnostics using the full stored reference masks,
not measured online query savings. Eleven new tests plus existing regressions
total99 passing tests. Syntax and default three-method dry-runs pass; standalone
lint/type/coverage tools are not installed, so those checks were not claimed.

Next revise the connection/context representation and distinguish true
installation restrictions from search-domain choices. Do not increase training
solely to address already optimal restricted-domain results, and do not claim
complexity from adding more redundant illustrative constraints.

## Integration After This Pilot

1. Saved-mask omission/accessibility analysis is complete. In the current cases
   the residual LS gap comes from fixed sides/anchors, not search within them;
   finite-boundary sensitivity and new generalization tests remain separate.
2. The bounded connection-band and station/side exclusion experiment is complete;
   extend to new infrastructure conditions and untouched geometries using explicit masks.
3. The fixed-rule new-width/turn test is complete and shows diversity can lose.
   Analyze omitted modes and failed corrections, then evaluate any adaptive
   source/target-query rule on another untouched set; obstacle changes remain pending.
4. Define a common canonical action/mask API before connecting existing MAPPO.
   Compare LS and fresh MAPPO with identical inputs, constraints and actual time.
5. Extend to multi-module/variable-K global selection with shared target identity,
   then assess selective response reuse and interface-only refinement.
6. Separately implement unfolded periodic scenes for genuinely repeated modules.

Feasible reuse range and near-optimal reuse range are separate questions.
Sampled-angle successes do not certify a continuous angular interval, and coverage
need not vary monotonically with angle. For a fixed map and feasible P, a valid
reference upper bound U gives `J* - J(P) <= U - J(P)`. Bounds from a different
candidate domain or constraint set cannot certify this statement.
