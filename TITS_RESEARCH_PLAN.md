# Research Development Toward a T-ITS Submission

Date: 2026-11-04, Asia/Seoul. Research proposal, not an implemented method,
novelty clearance, submission-ready manuscript or acceptance guarantee.
No new training, optimization or experiment is launched by this document.
Detection/tracking remain deferred. Historical experiments and defaults stay unchanged.

## Current Evidence And TR-C Readiness (2026-11-27)

The current consolidated reference is [RESEARCH_STATUS_20261127.md](RESEARCH_STATUS_20261127.md):
actual A/B source/target geometry, grids, optical and constrained MILP equations,
retrieval/transfer/selection rules, completed studies including negative findings,
and remaining TR-C evidence gaps. Earlier decisions below are historical, not the
current implementation inventory.

The 24-source occlusion/redundancy study and frozen-library low-K cost/critical
study have completed. The latter has 1,288 online rows, 182/184 common feasible
conditions and a +3.283 pp mean library-diverse advantage over geometry portfolio.
Fresh target raycasting and stored-result audits passed; 39 feasible full-domain
references still have gaps above 1%. These are reused development geometries,
not untouched real-road tests. Current results support a constrained reuse
prototype, not universal near-optimality, controlled speedup, automatic city-network
decomposition, minimum sensor counts, RL necessity or publication readiness.

The assessment recommends strengthening reference/quality-time comparisons,
isolating library size/layout/query effects, freezing held-out families, and
demonstrating a transportation infrastructure decision before adding RL/detection.
It is an internal evidence-based assessment, not a journal verdict or a new
experiment authorization. No new solve/training is launched by this update.

### Implemented Next Step: Fixed-Q Ablation

The subsequent [mechanism protocol](LIBRARY_MECHANISM_PROTOCOL.md) and
`library_mechanism_study.py` implement source-count/layout-count/search controls
on the existing development maps. Actual Q=256 is independent of library pool
size; explicit geometry fill makes this a disclosed hybrid ablation. The frozen
config also reserves eight additional geometry descriptors, but this runner
cannot evaluate them. Formal performance results remain TBD; only functional
verification is authorized/run in this implementation step. Historical statements
below about missing query-budget machinery describe their earlier dates.

## Current Decision: Fixed Comparison First (2026-11-19)

The current journal-development discussion targets TR-C. This filename and the
earlier T-ITS proposal remain for continuity; neither journal suitability nor
novelty has been established. The immediate decision is to freeze the non-learned
A/B reference and compare source reuse against strong geometry-only controls,
then expand only the aspects justified by the observed limitations.

[LIBRARY_COMPARISON_PLAN.md](LIBRARY_COMPARISON_PLAN.md) records the frozen
sources, K1/2/3, candidate/target grids, physical model, methods, interpretation
rules and gates for the next experiment. A and B are separate reuse tracks, not
competing algorithms. The completed November 18 comparison is retained as the
development reference; this decision launches no new run and is not a
preregistration of those already observed outcomes.

Do not simultaneously add source maps, layout alternatives, obstacles, costs,
constraints and RL. Source/layout and query-budget ablations remain possible
next studies, not approved execution specifications. When varying library size,
fix physical query budgets independently of pool size and disclose any hybrid
filling rule. Multi-source selection and that explicit-budget runner extension
are not implemented by this document. New factor levels, target manifest and
execution command remain TBD; untouched families must be declared before use.

Subsequent user-authorized execution reproduced the fixed four-map comparison
under `results/library_query_controls_repeat_20261119/`. All 408 raw and 12 summary
rows match the November 18 reference, with 131 preflight tests and all final
validation checks passing. No interim logs/outcomes were inspected. This repeat
does not add independent test maps or evidence of library-specific superiority;
the expansion plan remains unchanged.

## Current Direction And Reserved Learning Option (2026-11-18)

This decision supersedes the source-free research priority and gate order below,
not their experimental findings. The earlier plan remains a historical record.
The current priority is to build and test two reusable deployment libraries:

- **Track A, whole topology:** retrieve and transfer alternative deployments
  between roads with the same topology and different geometry.
- **Track B, components:** store deployment alternatives for road components and
  connection interfaces, then transfer and jointly select them on a new network.

First establish the benefit of non-learned retrieval, geometric transfer and
composition under basic physical/installation rules. Network-wide sensor
allocation and additional operational constraints follow. Geometry-only,
full-candidate and existing LS/MAPPO methods remain controls. Source reuse and
learning are hypotheses, not established advantages; retain all negative results.
The old partial-incidence bound proposal remains an optional supporting route,
not a prerequisite for building these libraries.

**Reserved extension, not implemented or scheduled:** learn from the stored
library itself, in either track. Candidate uses are:

1. Learn geometry-conditioned retrieval/ranking of source deployments or modules.
2. Learn a proposal model for road-relative deployment alternatives or sensor
   allocation, including imitation of recorded feasible deployments.
3. Learn a coverage/interaction surrogate to prioritize candidates for exact
   target-map evaluation.

No model family is selected yet. Library-trained inference, optional target-map
fine-tuning, and fresh per-map LS/RL refinement are distinct experimental arms.
This option neither requires MAPPO nor replaces the initial non-learned tests.
Source masks are labels for their original scenes, not ground truth on a new map.
Time-limited solutions are feasible examples, not automatically optimal labels;
retain the objective, constraints, domain, bound/gap and provenance of each.

For this extension, split source/train, development and untouched test geometry
families before fitting or tuning. Mirrors and variants of a family must not
leak across the declared split. Do not use held-out target results to train or
choose source candidates. Compare with the same library without learning and
with source-free controls; count offline library/training costs, online retrieval,
inference, transfer, refinement and verification separately, including amortization.
Final feasibility and quality require the common target-map physical evaluator;
learned predictions alone are not feasibility or optimality certificates.

Recording this retained learning option does not launch training or change
existing default settings or historical results. Source-builder implementation
progress is recorded separately below.

### Offline Storage Contract

[LIBRARY_STORAGE_PROTOCOL.md](LIBRARY_STORAGE_PROTOCOL.md) specifies the first
shared A/B implementation: maximum unweighted union coverage under at-most-K,
actual sensor counts, monotone best feasible frontiers, alternative layouts,
finite-domain bounds and minimum-count brackets. Do not confuse the union of
all eligible masks with 100% coverage or infer minimum count from a flat curve.
The first common backend is bounded exhaustive enumeration or greedy/provided
solutions with analytic bounds, not a new MILP solver integration. Source-only
storage validation is distinct from retrieval/transfer and network experiments.

Initial source builders now exist as `library_topology.py` and
`library_components.py`. Small K1/2/3 source runs are stored under
`results/library_ab_smoke_20261118/a_v2` and `b`, with exact finite-domain
enumeration and fresh saved-layout raycast checks. These are not completed A/B
transfer experiments by themselves or a full-size near-optimal source library.
Dedicated target adapters and their separate development checks are described
below; automatic retrieval and learned library use remain deferred.

### First A/B Target Checks (2026-11-18)

`library_transfer_study.py` completed four development targets, K1/2/3 each,
under `results/library_ab_transfer_20261118/`. A maps the 7m T source to two
width/junction variants by fractional installation coordinates. B maps the same
80m component twice to a 160m corridor with 45/60 degree turns. B is not yet a
heterogeneous module library or arbitrary intersection-network decomposition.
The pool recombines individual poses from stored alternatives under a global
at-most-K budget; no module quotas or additional operational constraints apply.

Pool choices are frozen before full-reference visibility is evaluated. All
reference and pool enumerations completed in their deliberately small domains.
Ten of twelve conditions matched the full finite-domain optimum. The 12m A
target lost 2.855 percentage points at K1 and 0.097 points at K3. Candidate pose
queries fell from 30/25/48/48 to 11/8/20/20 respectively. These are candidate
query counts, not controlled timing speedups or generalization evidence. The
original sources remained unchanged; 78 unit/regression tests passed. README
contains all twelve results, reproducible commands and artifact links.

The equal-query source-free controls are now completed below. Remaining gates
are broader source geometry and layout diversity, untouched target families,
and repeated cold-start timings that include loading/projection plus amortized
offline cost. Keep the observed A transfer losses and earlier negative source/
MAPPO findings. Additional constraints and LS/RL refinement follow these checks;
neither library novelty nor T-ITS readiness is established by this smoke test.

### Matched-Query Library Benefit Check (2026-11-18)

Protocol: [LIBRARY_QUERY_CONTROLS.md](LIBRARY_QUERY_CONTROLS.md). Results:
[condition table](results/library_query_controls_20261118/report/condition_summary.csv)
and [validation/summary](results/library_query_controls_20261118/report/summary.json).
Four known development targets, K1/2/3, four deterministic methods and 30 random
seeds produced 136 map/arm evaluations and 408 K rows, plus 12 full references.
The sources, target domains and library query counts (11/8/20/20) stayed fixed.
All query plans precede physical queries; every arm uses the same exact pool
selection; full-reference evaluation follows all deployment decisions.

Library-vs-control wins/ties/losses across the 12 conditions:

| Control | Wins / ties / losses | Mean library minus control |
|---|---:|---:|
| Geometry single ranking | 3 / 8 / 1 | +0.139 pp |
| Geometry greedy-union ranking | 2 / 9 / 1 | -0.174 pp |
| Geometry alternative portfolio | 2 / 8 / 2 | +0.122 pp |
| Random mean, seeds 0..29 | 11 / 0 / 1 | +1.126 pp |

These are equal-map/K descriptive summaries, not inferential comparisons.
Library, union and portfolio each match the full finite-domain optimum in 10/12
conditions; single ranking matches 9/12. All six B conditions tie across all four
deterministic methods. Source reuse is therefore not necessary to attain this
quality on these B cases. A includes both benefits and losses, including the
previous 2.855 pp loss at 12m/K1. A random-only comparison would overstate benefit.

**Decision:** candidate reduction works here, but consistent library-specific
quality superiority is not established. Do not promote RL, operational constraints
or broader deployment claims from this result. Next separate source/layout
diversity and query-budget effects with fixed rules, then evaluate untouched
geometry families. Retain all strong controls and negative transfer examples.
Use realistic variation to test the hypothesis, not to select favorable cases.

All 420 frontier objectives and tie-selected IDs passed an independent integer-
bitset exhaustive check of saved masks. Per-worker saved layouts passed fresh
raycast verification; previous library/full outputs reproduce unchanged.
Final regression: 125 tests passed, no failures, errors or skips. README's 84
comparison values were checked against the generated CSV.
Equal q is not equal rays, CPU work or runtime: geometry plans evaluate all
unoccluded pose-target pairs and portfolio enumerates predicted subsets.
No controlled speedup, offline-cost amortization, real-road generalization,
minimum network sensor count or publication-level novelty follows from this check.

## Infrastructure Review (2026-11-09)

[RSU infrastructure and wireless reward review](RSU_INFRASTRUCTURE_REVIEW_20261109.md)
selects recent 2024-2026 primary studies and official deployment reports. Proposed
priorities are route-level geometric gaps, declared occlusion/pose-error scenarios,
and approved sites with power/backhaul paths and shared costs. Wireless fairness,
bottleneck service and reliability objectives are transferable structures, not
LiDAR physics or evidence that MAPPO is better. Numeric service requirements remain
unspecified. This review changes neither code nor historical experiment settings;
Gate 2 remains the next method-validation gate, not a completed implementation.

## Gate 1 Update (2026-11-09)

The original plan below is preserved. Gate 1 is now implemented and executed;
the method/bound/network proposals in later gates remain unimplemented.
Protocol: [GEOMETRY_MECHANISM_STUDY.md](GEOMETRY_MECHANISM_STUDY.md).
Results: [468-worker report](results/geometry_mechanism_20261109/REPORT.md).

Static pair-aware ranking and the same-schedule no-feedback control were compared
with existing adaptive/single rankings and full on the same 12 development
conditions, q=1/2/4, with three timing repetitions. All workers and 75 regression
tests passed. No source assets, detector or MAPPO were used.

**The measured-feedback benefit is not supported in these cases.** Adaptive and
no-feedback return the same sensor IDs and J in every case/budget, even though
query pools differ in 8/12 q2 and 10/12 q4 cases. Static pair-aware ranking beats
adaptive in 4/12 q2 cases and ties eight; q2 full-reference matches are 10/12 vs
6/12. At q4 all three match full in 12/12 cases but are slower: mean per-case
median cold-ready 3.127/3.147/3.133 s for adaptive/static-pair/no-feedback vs
2.879 s full. Static-pair also loses to single ranking in two q1 cases.

Consequences: preserve these negative controls, remove an established-feedback-
advantage claim, and do not rename pair-aware ranking as novelty. Continue with
the source-free research question, not an obligation to retain adaptivity or RL.
The next bounded implementation is Gate 2: test soundness, tightness and cost of
partial-incidence bounds on tiny enumerated problems. If these are vacuous or
expensive, revise the proposal before launching a larger network experiment.
No global quality certificate, network scalability or T-ITS readiness is established.

## Decision

Target question:

> Can geometry- and constraint-aware selective visibility evaluation produce
> high-quality feasible roadside LiDAR deployments on assembled road networks,
> with less online computation than full visibility construction, while exposing
> the remaining uncertainty in solution quality?

Working title: Query-Efficient Constrained Roadside LiDAR Deployment on Road Networks.

The main problem is network deployment, not candidate ranking on one corridor.
The proposed technical core is a useful, inexpensive representation of unresolved
visibility and its effect on global feasibility/quality, plus a query policy that
uses it. Generic active sampling, a master MILP, multiple constraints, a GNN,
MAPPO, and a new name do not by themselves establish novelty.

Geometry-only remains the primary path. Source libraries are optional priors or
warm starts, not required evidence of success. Reintroduce source transfer as a
main contribution only after it provides a repeatable incremental benefit over
source-free selection with its loading, matching and construction costs counted.
Do not discard the existing libraries; preserve them as controlled alternatives.

## Venue Fit And Closest Overlap

The official [T-ITS scope](https://ieee-itss.org/pub/t-its/) includes sensing,
optimization, infrastructure design, modeling and simulation. It explicitly asks
infrastructure papers to address their impact on the transportation process.
This supports investigating deployment design, but generic geometric coverage
alone leaves the transportation benefit insufficiently demonstrated. Detection
is not prescribed by that scope; omitting it does not guarantee journal fit.

Primary sources checked on the date above; this is a targeted review, not an
exhaustive search. Publisher previews/author abstracts cannot support claims
about undisclosed implementation details.

| Work | Inspected evidence | Consequence for our claim |
|---|---|---|
| [Vijay et al., ITSC 2021](https://arxiv.org/abs/2110.01251) | Author abstract: raycasting and linear optimization with cost, coverage and redundancy | Adding these constraints is not new by itself. |
| [Jiang et al., ICCV 2023](https://arxiv.org/abs/2310.07247) | Author abstract: learned perception evaluation and greedy perceptual gain | Marginal contribution or an evaluator alone is insufficient differentiation. |
| [Chen, Zheng and Tan, TR-C 2024, 104838](https://www.sciencedirect.com/science/article/pii/S0968090X24003590) | Publisher-indexed abstract/introduction: costly stochastic detection evaluation, chance constraints, GPR-AKG sampling | Constraint-aware active evaluation for roadside placement already exists; inspect the full method before claiming superiority or reproducing it. Full text was not obtained here. |
| [Jiang et al., IV 2025](https://arxiv.org/abs/2504.06772) | Author abstract: entropy-guided visibility surrogate for detection evaluation | Fast visibility-related evaluation is also established. |

Our proposed distinction is deterministic geometric incidence uncertainty tied
to network-level coupled constraints and a checkable quality bound, rather than
a whole-layout detection predictor. This is a hypothesis to investigate, NOT a
claim that no existing paper has done it. Read neighboring visibility, lazy
optimization, sensor selection and decomposition work before finalizing novelty.

## Current Evidence And Missing Evidence

The [equal-query study](results/geometry_budget_20261104/REPORT.md) completed
264 workers on six known development geometries and two profiles, fixed K=2.
At four queries/site, adaptive matches the full finite reference in 12/12 cases,
static single-sensor ranking in 4/12, and random in 7/60 case-seed outcomes.
Adaptive and full mean cold-ready times are 3.1244 and 2.8842 seconds respectively:
query efficiency is supported here; full-relative speed advantage is not.
The exact command, frozen inputs/source and audit are linked from the report.

Reuse: the geometry/visibility engine, installation/coverage evaluator, query
accounting, packed candidates, finite exact references, paired reporting and
independent audits. Keep historical J definitions and outputs intact.

Missing: isolated feedback contribution, arbitrary network geometry, variable
global K/allocation, useful global quality bounds under partial visibility,
untouched geometry-family evaluation, realistic network assets, and a controlled
end-to-end advantage. The current full reference is not a continuous optimum.

Targeted Scholar Evaluation judgment: method differentiation 2/5 and external
evidence 2/5; internal reproducibility is substantially stronger than external
validity. These are planning assessments, not reviewer scores or acceptance odds.

## Proposed Method And Proof Obligations

1. Define the full eligible candidate domain from geometry and installation
   rules before querying target visibility. Keep globally unique target IDs and
   sensor identities. A restricted library must not silently become the domain.
2. Maintain per candidate-target incidence bounds a_minus <= a_true <= a_plus.
   Range/FOV exclusion can set both to zero; an exact query resolves an entry.
   Unoccluded geometry may give an upper envelope, not a positive LOS certificate.
   Geometry edits require checking changed endpoints and relevant occluders;
   source masks or learned confidence are not automatically valid bounds.
3. Solve a global budget/allocation problem using the current information.
   Use verified deployments for feasible lower bound L, and a relaxation covering
   every eligible deployment for upper bound U. A restricted pool optimum is
   neither a global upper bound nor evidence of global infeasibility.
4. Query unresolved candidates/ray groups that can affect active constraints or
   improve the incumbent/bound. Geometry, constraint slack and neighboring
   coverage determine query priority. The exact score is TBD and must beat
   simple uncertainty-only and geometry-only rules at matched total work.
5. Re-evaluate final selected poses with the common physical evaluator. Report
   the actual finite-domain certificate and stop on U-L <= epsilon only when
   both bounds are valid. At a budget/time limit, report the remaining gap or
   unknown status; never claim epsilon-optimality automatically.

For maximization the target statement is L <= J_star <= U and, for a verified
returned incumbent, J_star-J_returned <= U-L. This is a standard bound argument,
not our novel theorem. The research must supply a sound and useful cheap bound
construction or query/composition result specific to this problem.

Pure nonnegative weighted union is monotone in incidence. The present objective
also penalizes overlap: substituting all upper visibility values into J is NOT
generally an objective upper bound. Lower/upper redundancy requirements and
single-sensor failure need separate interval reasoning and sensor identity.
Do not import a monotone-greedy approximation ratio for the complete constrained J.
If useful bounds remain too loose or cost as much as full raycasting, revise this
route rather than retaining a vacuous certification claim.

Network composition must count target multiplicity globally; summing module
coverage or subtracting pairwise overlap misses higher-order interactions.
The visibility interaction structure must include nearby nonadjacent roads,
not just road-topology neighbors. Include capacity/spacing conflicts globally.
Representation on a sphere/cylinder remains optional; it must not wrap physical
rays across unrelated boundaries or discard depth and occluder identity.

## Transportation Evaluation Without A Detector

Specify a service contract before final experiments: sensor/installation budget,
approved mounting sites and heights, selected critical regions, required
geometric coverage/redundancy and any declared failure contingency. Use only
constraints that can be justified and vary them in sensitivity analysis.
Current numerical thresholds are illustrative, not validated RSU standards.

Report weighted geometric service coverage, cost and sensor count, critical-zone
and connection-region deficits, and contiguous geometric visibility gaps along
declared lane trajectories. Gap length can be converted to time only under an
explicit speed/trajectory assumption. Test trajectory/sample-height resolution.
These are geometric observability proxies, NOT detector recall, tracking
continuity, localization accuracy, crash reduction or guaranteed driving safety.

Use unused realistic multi-intersection geometry with validated 3D occluders,
installation sites and provenance; a 2D OSM polyline alone is not a 3D scene.
Missing building heights and mounting data need explicit assumptions/sensitivity.
Static visibility is the first scope; dynamic traffic occlusion is an optional
later scenario study, not already implemented or a substitute for detection tests.

## Ordered Gates

| Gate | Work | Evidence required before expanding the claim |
|---|---|---|
| 1. Mechanism | Add static pair-aware ranking and a no-feedback version of the same conditional schedule | Matched domains, objectives and pose-query budgets; separate pair scoring from measured-mask feedback; include all failures and negative results. |
| 2. Bounds | Implement partial-incidence bounds on tiny problems before optimizing performance | Exhaustive completions/subsets verify validity; test overlap penalties, lower/upper redundancy and failure identities; measure bound tightness and construction cost. |
| 3. Network | Extend from the corridor to assembled paths/branches/intersections and variable global K | Shared scene, unique targets, physical interaction/conflict checks; small exact references; demonstrate when global allocation matters. |
| 4. Scale | Vary candidates, targets, mesh/occluder complexity and network size independently | Quality-query and quality-time curves, feasibility and memory; include preprocessing, bounds, selection, optimization and final verification under matched timing boundaries. |
| 5. External evidence | Freeze method/parameters, then evaluate untouched geometry families and realistic network assets | Family-level splits, justified service conditions, paired comparisons and uncertainty at the map-family level; mirrors/seeds/profiles are not independent maps. |
| 6. Manuscript | Claims-to-evidence table and full closest-work comparison | Distinguish algorithmic contribution, conditional proofs and experimental evidence; quantify limits rather than describing every existing component as new. |

Gate 1 controls must be explicit: static pair-aware ranking freezes scores before
actual queries. The no-feedback schedule uses the same rounds, tie rules and
pair constraints as adaptive, but supplies unoccluded predicted masks instead of
measured partner masks while planning. Freeze its entire query plan before any
actual raycast. Virtual planning evaluations must not be counted as real queries;
both costs are recorded. A fixed pair ranking alone does not isolate feedback.

If feedback does not improve the stronger controls, drop the feedback-benefit
claim. Do not change test cases or thresholds just to recover a win. If source
priors are later added, compare with geometry-only and account for offline
construction/amortization. If learned query policies are added, train across
separate maps and compare with simple deterministic rules; fresh per-map MAPPO
is not amortized transfer. Neither learning nor a theorem is an automatic venue
requirement or an acceptance guarantee.

Freeze final test families, primary metrics, practical quality tolerance, budgets
and statistical units before testing. Select sample sizes from development
variability and desired precision, not an invented journal minimum. Keep
pose-query counts, actual rays, CPU work and end-to-end latency distinct.

Original immediate implementation was Gate 1; it is complete with the limitations
reported in the 2026-11-09 update. Gate 2 is next, not yet implemented. A full
T-ITS-strength contribution remains a research target, not the repository status.
