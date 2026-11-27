# Research Directions From the Existing Deployment Pipeline

Date: 2026-10-22, Asia/Seoul. Exploratory scoping review and research design,
not a systematic review, novelty clearance, implemented algorithm or new result.
Detection/tracking remain deferred. Existing code, thresholds and results are
unchanged. This note extends [the earlier review](RSU_RESEARCH_REVIEW_20261020.md)
with the completed connection/constraint experiments.

## What the Existing Results Actually Establish

Existing assets include source pose/geometry archives, road-relative transfer,
multiple alternative poses, shared static raycasting, constrained J evaluation,
LS/fresh MAPPO implementations, packed full-grid references and independent audits.
The 410xxx source library is not a collection of universally certified optima.

The fixed-rule new-geometry experiment has24 conditions and168 saved layouts:
[report](results/connected_constraint_holdout_20261021/analysis/REPORT.md).

- Top-source pools match the global finite-grid J in17 conditions; diverse pools
  match in15. Diversity wins7 pairwise comparisons, ties11 and loses6.
- Both source-pool selectors find feasible layouts in all24 conditions.
  Independent transfer violates mounting exclusions in12; these are not12
  failures caused solely by road curvature.
- Every source-pool optimum is independently checked. Thus its raw shortfall
  is a pool restriction, not an uncompleted search within that pool.
- Global and one-per-module reference J are equal in all24 cases. Sensor-count
  allocation between modules is not yet demonstrated to be difficult here.
- Diverse+LS recovers three cases and matches18 optima. The remaining six
  require separating unreachable designs from local-search barriers.
- No timing speedup, policy-transfer benefit, real-road generalization or
  continuous/global-city optimum follows from these measurements.

These negative comparisons motivate a method for preserving *composition value*,
not another unconditional diversity heuristic or more training on the same cases.

Update after the first diagnostic step: the saved-mask analysis is now complete
([report](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/REPORT.md)).
All48 saved LS outputs already maximize J in their original side/anchor domain.
The six diverse+LS gaps are therefore accessibility restrictions, not unexploited
local search in that domain. Minimum-addition optimal witnesses expose a pitch
alternative discarded by height/yaw-mode compression, while top-score pools
miss high-mount oblique alternatives. These are hindsight diagnoses, not a new
selection rule or a general impossibility result for local correction.

The first source-response baseline is now implemented and evaluated on those
same observed24 development conditions ([results](results/connected_response_study_20261022/REPORT.md)).
At four alternatives/site, source-only marginal-response regret compression
improves geometric diversity in three raw cases (+1.4 J each), ties21, and loses
none. Raw optimum matches remain15/24; after LS, both rules match18/24 and have
identical J in all24 cases. The48 full/quota references,72 pool optima and216
saved deployments passed independent auditing. This is not new held-out evidence,
a learned transfer model, or a demonstrated advantage over diversity+LS. Source
response fidelity by itself does not establish preservation of composition value
under changed target geometry and coupled constraints.

## Prior Work That Limits Easy Novelty Claims

Only primary manuscripts, proceedings pages, author abstracts and publisher
previews support this table. Version/publication qualifications are explicit.

| Work | Evidence Inspected | Consequence |
|---|---|---|
| Jiang et al., ICCV2023 | [Full author manuscript](https://arxiv.org/html/2310.07247v1): perception predictor and incremental perceptual-gain placement | A response predictor or marginal-gain selection alone is not a new contribution. Their task is detection, unlike our static coverage. |
| Chen, Zheng and Tan, TR-C2024,104838 | [Publisher abstract/preview](https://www.sciencedirect.com/science/article/pii/S0968090X24003590): chance constraints and GPR-AKG sampling for costly stochastic roadside placement evaluation | Adding constraints and active sampling is not sufficient differentiation. Full text was not accessed in this review. |
| Mirrokni and Zadimoghaddam,2015 | [Author abstract](https://arxiv.org/abs/1506.06715): randomized composable coresets for coverage/submodular optimization | Selecting local representatives and optimizing their union is already an established abstraction. Its assumptions differ from changed geometry and coupled service constraints. |
| Balkanski et al., ICML2016 | [Proceedings](https://proceedings.mlr.press/v48/balkanski16.html), [full manuscript](https://web.cs.ucla.edu/~baharan/papers/balkanski16learning_long.pdf): reusable representative sets and task-specific subset selection | Learning a candidate dictionary for downstream objectives is not itself a new mathematical model. |
| Satsangi, Whiteson and Oliehoek, author version2020 | [arXiv v2](https://arxiv.org/abs/1602.07860v2): greedy sensor selection with cheap anytime confidence bounds | Bound-guided selective evaluation already exists; a contribution needs a specific useful geometry-valid bound construction. |
| Wang et al., author version2024 | [Full arXiv v2 manuscript](https://arxiv.org/html/2310.12387v2): transformer/RL sensor relocation improvement heuristics | Learned sensor relocation is not new. Fresh per-map PPO is not evidence of an amortized reusable policy. |
| Xiong et al., VF-Plan,2025 preprint v1 | [Full manuscript](https://arxiv.org/html/2503.01562v1): visibility fields, candidate reduction and overlap-connected viewpoint planning | Visibility fields and overlap graphs already exist, although static scanning differs from roadside network transfer. |
| Lucic, Ghazzai and Massoud,2020 author manuscript | [Abstract](https://arxiv.org/abs/2007.06043): coverage optimization with energy and throughput capacity constraints | More infrastructure constraints make a stronger model, not automatically a new method. |
| Jiang et al., IV2025 | [Author abstract](https://arxiv.org/abs/2504.06772), marked accepted at IV2025: entropy-guided visibility surrogate | Faster roadside evaluation through a surrogate is also prior art. |
| Agarwal and Balkanski, NeurIPS2024 | [Conference paper](https://papers.neurips.cc/paper_files/paper/2024/file/19cdab1dee61d55158cf106244ceab08-Paper-Conference.pdf): prediction-assisted dynamic submodular maximization | Learning-assisted updates are established; changing geometry/incidence is not the same as predicted candidate insertion/deletion. |

These are closest conceptual overlaps, not evidence that a proposed method is
absent elsewhere. No first-ever claim is justified by this search.

## Route A: Libraries That Preserve Conditional Composition Value

Question: With a fixed library/query budget, which alternatives must survive
compression so that a new network can still form a good feasible deployment?

Replace source-only score and height/yaw diversity with a library trained or
constructed against *downstream composition regret*. Context includes neighbor
coverage, connection geometry, approved sites, sensor budget and requirement
slack. Preserve branch/side allocation modes instead of forcing incompatible
deployments into one-to-one pose interpolation.

For the current objective, let a_pj be candidate p's binary visibility and d_j
the density from the other sensors. Its exact marginal objective is:

```text
Delta J(p | d) = sum_{j:d_j=0} w_j*a_pj
                - lambda*sum_{j:d_j>=1} a_pj
                - beta*cost(p).
```

This formula assumes weighted union minus excess observations and additive cost;
hard constraints still need a separate feasibility test. The formula is not new.
It explains why a high standalone score or geometric distance is insufficient.

A concrete representation to investigate is a library of conditional response
envelopes: alternatives expose interior contribution, boundary density support,
cost/K and constraint-relevant signatures. Different external coverage patterns
can make different alternatives optimal. Geometry transfer must explicitly
identify which support is a prediction and which has been revalidated.

Novelty target: a compact road/connection-relative representation and selection
algorithm that retains useful alternatives under geometry and requirement
changes, beyond generic dictionary/coreset selection.

First experiment: replay stored masks offline with fixed pool sizes. Compare
top-score, geometric diversity, random, source-mask diversity, a standard
two-stage representative-selection baseline and context-conditioned selection.
Use development contexts for construction; reserve new geometry families for
testing. Score joint feasible quality and candidate/ray budgets, not isolated
candidate accuracy. Teacher target optima are permitted only offline in training.

Stop or revise this route if a simple source-mask baseline matches the proposed
representation, or if generating the context consumes almost the full target
visibility archive. A target mask used for candidate selection counts as a query.

## Route B: Decide What Can Be Reused and What Must Be Recomputed

Question: Can geometry provide sound reuse tests and useful uncertainty bounds
so that only decision-relevant target rays need fresh computation?

Transfer is a prediction, not ground truth. Keep three types of incidence:
certifiably unchanged, physically impossible, and unknown. Learned confidence
may prioritize queries, but is not a geometric certificate. For a visibility
claim, account for changed ray endpoints, FOV/range and every relevant occluder.
Arbitrary width/curvature changes need not preserve any source binary mask.

Maintain a verified feasible incumbent L and a valid relaxed upper bound U.
Query uncertain relations that might change feasibility or improve the bound;
stop at a specified additive U-L only when the bounds cover the declared domain.
A bound over the restricted library does not certify omitted candidates.

Novelty target: an inexpensive road-relative validity/uncertainty representation
that makes this process useful under changed geometry, not the standard
branch-and-bound or active-learning principle itself.

Important: J penalizes additional overlap. It is not valid simply to substitute
all upper density bounds into J and call the result an upper bound. Optimize
each admissible density interval, or use a sound relaxation. Handle multiplicity,
N-1 sensor identity and lower/upper service constraints explicitly.

First experiment: compare full recomputation, uniform queries, geometry-only
query ordering, target-only active sampling and source-assisted selection at
the same verified quality target. Include bound construction, correspondence,
source loading and final independent verification in the accounting.

Stop or revise if constructing useful bounds costs as much as direct evaluation,
or most intervals stay too loose to prune anything. Short modules and global
width changes are especially important adverse cases; boundary halos may cover
nearly the whole module.

## Route C: Network Composition With Variable Local Budgets

Question: Can local alternatives be combined under a global sensor/cost budget
without losing cross-boundary coverage, useful redundancy or joint feasibility?

Represent multiple local alternatives for each budget K_m, including different
branch/side allocations. Optimize selections and K_m jointly. All target points
have global identities; sum actual sensor incidence before applying coverage
or redundancy requirements. Do not just sum isolated module scores.

Store connection-conditioned response signatures, then revalidate changed and
cross-module support in the assembled geometry. The interaction graph must
include spatially nearby roads within sensing range, not only adjacent roads
in the traffic topology. Pairwise overlap subtraction is not exact for triple
overlap. Shared hubs, pole capacities and failure identities remain global.

Novelty target: a transferable interface representation and an allocation/
composition algorithm with measured loss/work advantages over solving each new
assembled network from scratch. Using a generic master MILP is not alone novel.

First experiment: extend from two to three modules with variable allocation;
then evaluate larger chains and branch networks. The present curved-corridor
engine cannot already assemble arbitrary junction graphs. Shared geometry,
occluder union and global-target ownership require new implementation/validation.
Use cases where allocation actually matters rather than assuming it does.

Compare independent equal allocations, the same global optimizer without source
information, pose-only libraries, conditional-response libraries and small exact
references. Increase library budgets equally across competing methods.

Stop or revise if equal local allocation is consistently optimal, independent
module solutions already suffice, or global coordination only helps because the
baseline omitted the same final constraints.

## Route D: Learn a Reusable Correction or Query Policy

Question: Can one trained policy decide where to query, replace a candidate or
coordinate multiple changes across unseen assembled geometries?

This differs from fresh PPO per target. Use graph/port/constraint features and
variable-size inputs. Consider actions for candidate replacement, module-budget
reallocation, coordinated repair and selective evaluation, not only small pose
increments. Enforce the same declared final constraints as all baselines.

Novelty target: a geometry/requirement-conditioned policy that amortizes online
work or resolves demonstrated coordination barriers. MAPPO is an optional
implementation, not the definition of the contribution.

First experiment: inspect the six diverse+LS misses with stored masks. Separate
inaccessible poses, one-agent local barriers and missing candidates. Compare
matched-action LS, multi-start/pair-exchange search, a simple supervised ranker,
a bandit and a trained reusable policy. Include all query and repair costs.

With only two sensors, exhaustive pair replacement already solves the finite
joint problem; it is not evidence of an RL coordination advantage. Larger
networks and repeated target tasks are needed to motivate amortized learning.

Stop or revise if a simple ranker or matched-action search performs equally well,
or if the claimed benefit requires target-specific retraining that removes the
online saving after training cost is included.

## Recommended Paper Scope

Use Route C as the main problem and Route A as the main method. Add a bounded
part of Route B only if cheap sound reuse tests are practical. Keep Route D as
a later comparison or separate contribution rather than requiring every module
to use RL.

Proposed question, not a demonstrated result:

> Can connection- and constraint-conditioned deployment libraries preserve
> composition quality on unseen road networks with fewer target evaluations
> than full cold-start placement optimization?

The paper needs an identifiable representation/algorithm, not just library,
interpolation, GNN, MILP and MAPPO presented as a list of assembled components.

The sphere/cylinder idea can be a representation ablation: a direction-plus-depth
boundary signature with labeled ports, compared against simpler road-relative
coordinates. It must not identify unrelated boundaries or bend physical rays.
A direction-only spherical image loses depth. Encoding alone is not yet a
defensible novelty claim, and actual periodic wrapping is still unimplemented.

## Mathematical and Experimental Targets

For a fixed true-feasible full domain D and fixed feasible library domain Q,
with returned P_out in Q:

```text
J(P_D*) - J(P_out)
  = [J(P_D*) - J(P_Q*)] + [J(P_Q*) - J(P_out)].
```

This standard identity separates omission and optimization error. For current
raw pool selection, the second term is zero within numerical tolerance. LS may
leave Q, so analyze its reachable refinement domain separately.

If a surrogate has uniform error <=epsilon over the SAME true-feasible Q and
the optimizer's certified additive error is eta, the second term is bounded by
2*epsilon+eta. This is a standard conditional error bound, not our novelty.
Average prediction error and fresh MAPPO do not certify those assumptions.

A worthwhile new theorem would establish a checkable preservation condition for
the proposed response compression or a tight geometry-dependent reuse bound.
Any sum of per-module errors also requires a feasible global replacement
argument; coupled overlap, spacing and failure constraints cannot be ignored.
Do not import monotone-greedy guarantees to the whole current constrained J.

Evaluation requirements:

1. Diagnose existing saved failures without changing them or fitting a final
   rule on them. These cases are now development data.
2. Prespecify train/development/test geometry families; rotate/mirror copies are
   not independent new maps. Vary width, angle, occluders and approved poles
   independently before attributing effects to one geometric factor.
3. Keep fixed query and actual-time comparisons separate; report feasibility,
   constrained J regret, boundary service, sensor count and offline amortization.
4. Normalize objective terms or use declared physical quadrature weights before
   cross-map learning; keep historical raw-J experiments unchanged. Required
   overlap and penalized excess need an explicit operational interpretation.
5. Keep equal candidate/action/constraint information for baselines. Test learned
   policies against simple conditional selection and coordinated search.
6. Use exact same-domain small instances for certification; use honest bounds
   and best-known labels on large instances. A new road-network benchmark is
   supporting evidence, not a replacement for a method contribution.

No numeric acceptance targets or field requirement thresholds are established
here. Fix practical quality tolerances and query/time budgets before the next
evaluation, not after observing which method wins.

## Search Record and Limitations

Search date2026-10-22. English technical searches; no lower date bound. Primary
arXiv, PMLR, NeurIPS, author and publisher pages were checked. This is not a
Scopus/Web of Science export, exhaustive review or patent search. Versions are
consolidated by arXiv ID/title. Secondary reviews were discovery leads only.

Exact search strings in five batches of four:

```text
roadside LiDAR placement optimization transfer geometry library connected road network
sensor placement coverage composable coresets distributed maximum coverage geometric
sensor placement certified visibility bounds lazy evaluation ray tracing optimization
transfer learning sensor placement reinforcement learning constraint geometry
"composable core-sets" "submodular maximization"
"roadside LiDAR" "chance-constrained"
"sensor placement" "lazy" "visibility"
"roadside" "LiDAR" "network" "placement optimization"
"roadside" "LiDAR" "decision transformer" placement
"CRSLP" LiDAR placement
"roadside LiDAR" "transfer" placement learning
"sensor placement" "two-stage" "submodular" learning
roadside LiDAR placement Zhao decision transformer optimization 2025
"LiDAR" "placement" "decision" "transformer"
"Learning Sparse Combinatorial Representations" two-stage submodular Balkanski
"submodular" "lazy" "oracle" "confidence" optimization
sensor placement coverage spatial decomposition boundary interfaces optimization
view planning visibility fields overlap VF-Plan 2025
roadside LiDAR placement energy throughput capacity Lucic Ghazzai Massoud 2020
"learning augmented" "submodular" "dynamic" Balkanski
```

An additional primary dataset lead is
[CRSLP, SSRN2026](https://papers.ssrn.com/sol3/Delivery.cfm/616d13c2-95c9-43f9-ba39-06cadf7a1c75-MECA.pdf?abstractid=6816004&mirid=1&type=2)
and its [author project page](https://yangma93.github.io/CRSLP-Benchmarking-cooperative-roadside-lidar-placement/).
It is a preprint benchmark involving perception; no data were downloaded,
license/reproduction audit performed or detection evaluation added here.
Some publisher/full-PDF fetches were blocked; inaccessible content was not
treated as read. Negative search results do not prove absence of prior work.
