# RSU Coverage Research Review

Date: 2026-10-20. Type: bounded scoping review and read-only implementation audit.
Scope: static geometric coverage, installation constraints, transfer and network
composition. Detection and tracking remain outside the proposed first experiment.
The proposals below are not implemented results or established novelty claims.

## Research Question

Can a library of geometry-relative candidate responses be reused and composed
on a different road network, while recovering feasible coverage quality with
fewer target-map ray queries than solving the new network from scratch?

The most promising contribution to investigate is the representation and
verification of reusable responses, including cross-module effects. MAPPO is
one possible refinement method; its advantage must be evaluated separately.

## Findings From the Current Implementation

| Finding | Evidence | Consequence |
|---|---|---|
| Cost budget is nonbinding in the infrastructure/resilient profiles | Fixed K, height <=5, cost per sensor =10+z, budget =15K | Every height-feasible placement already satisfies this budget. This is a structural fact, not only a property of the observed solutions. |
| Pole capacity duplicates another constraint | Minimum XY separation is 3m; pole identity is exact XY equality | Positive separation already excludes two sensors at the same pole. These are not independent sources of difficulty. |
| Some service thresholds are loose at the saved solutions | Global coverage >=75%, branch coverage >=60%, global overlap <=90%; saved global coverage is 98.990-99.903%, branch coverage >=97.530%, overlap 24.175-47.033% | These constraints are not tight at the reported endpoints. They may still reject intermediate proposals; rejection causes are not yet aggregated. |
| Required and penalized redundancy need a clearer application interpretation | J penalizes every observation beyond the first, including critical-region observations; critical overlap and N-1 protection are lower bounds | Once a threshold is satisfied, J can prefer reducing additional redundancy. This is a tradeoff, not an implementation contradiction. |
| Cost tradeoff depends on target sampling | `objective_from_density` sums covered targets, weighted coverage and excess observations, but subtracts unscaled monetary-style cost | Duplicating targets scales visibility terms but not cost. Changes in target density can change the effective preference. |
| Search accessibility differs from installation feasibility | 2m XY actions, per-sensor original-anchor radius 10m, physical action masks and infeasible-joint self-loops | A feasible final design need not be reachable from the transferred design under this search. Longer training cannot remove an action-space exclusion. |
| The policy is freshly trained and has fixed input dimensions for the current K/ROI/hub layout | Fresh `SharedMAPPO`; concatenated agent state and appended ROI/hub features | Current evidence concerns per-instance learning. It does not establish reusable geometry/constraint-conditioned policies. |
| Transfer still assumes matching branch/side structure and connected installation intervals | `prototype_segments.py` rejects disconnected segments and incompatible topology | More widths and shifted junctions do not test general road-network composition or changed installation topology. |

Code references:
[profile generator](rsu_coverage_study.py),
[objective](src/core/evaluation.py),
[requirements](src/core/coverage_requirements.py),
[refinement](src/methods/constrained_refinement.py),
[geometric transfer](prototype_segments.py).

### An Important Existing Tradeoff

The completed two-map check used K=4, resilient constraints, seed 0 and 4,096
MAPPO steps / 16 PPO updates. These are the saved best-J deployments:

| Target / metric | LS | MAPPO |
|---|---:|---:|
| T 420002: J | 31,353.75 | 31,085.75 |
| T 420002: global coverage % | 99.903 | 99.673 |
| T 420002: worst single-failure junction coverage % | 40.476 | 96.310 |
| Cross 420004: J | 39,756.10 | 39,264.75 |
| Cross 420004: global coverage % | 99.836 | 98.990 |
| Cross 420004: worst single-failure junction coverage % | 98.060 | 98.599 |

The T-junction LS solution is close to the configured critical-region failure
threshold of 40%. This suggests examining a coverage/cost/resilience tradeoff.
It does not establish MAPPO superiority: LS was optimizing the specified J,
and both methods need rerunning under any newly specified requirement.

Training rejection rates were 13.26% and 8.28% for the two MAPPO runs. These
rates alone do not show that rejection caused the performance difference.
Log rejection reasons and feasible-neighborhood sizes before changing training.

Source: [raw method results](results/rsu_coverage_development_20261020/raw_methods.csv).
Original execution command is in [RSU_COVERAGE_PROTOCOL.md](RSU_COVERAGE_PROTOCOL.md).
No new training, solving or raycasting was performed for this review.

## Closest Prior Work

Evidence below comes from primary papers or publisher previews. A preview
supports only the accessible claims; reported rankings are not independently
reproduced here. Versions of the same paper are treated as one work.

| Work / identifier | Inspected source and relevant contribution | Implication for our research |
|---|---|---|
| Vijay et al., ITSC 2021; arXiv:2110.01251 | [Author manuscript](https://arxiv.org/html/2110.01251v1): raycasting and binary optimization with cost, coverage and redundancy considerations | Adding these factors is not the contribution. |
| Jiang et al., ICCV 2023; arXiv:2310.07247 | [Author manuscript](https://arxiv.org/html/2310.07247v1): learned perception predictor and incremental perceptual-gain greedy placement | Predicting a response map or evaluating marginal gain alone is already close to existing work. Their detection objective differs from our geometric coverage. |
| Chen, Zheng and Tan, TR-C 2024; DOI:10.1016/j.trc.2024.104838 | [Publisher preview](https://www.sciencedirect.com/science/article/pii/S0968090X24003590): costly stochastic roadside LiDAR evaluation, chance constraints and GPR-based approximate knowledge-gradient sampling | Selectively evaluating promising placements with a surrogate is also prior art. We need a specific geometry-transfer/composition contribution. |
| Xiong et al., VF-Plan, arXiv:2503.01562v1, 2025 preprint version | [Full manuscript](https://arxiv.org/html/2503.01562v1): visibility fields, structural candidate reduction and overlap-connected static scanning viewpoints | A visibility field or overlap graph alone is insufficient differentiation. Indoor scanning/registration is different from roadside deployment transfer. |
| Ma et al., Tunnelling and Underground Space Technology, 2026; DOI:10.1016/j.tust.2026.107629 | [Publisher preview](https://www.sciencedirect.com/science/article/pii/S0886779826001872): curved-tunnel geometry, probabilistic grids, neural surrogate, greedy/genetic optimization | Curved geometry plus a learned surrogate is not an untouched direction. |
| Wang et al., arXiv:2310.12387v2, 2024 preprint version | [Full manuscript](https://arxiv.org/html/2310.12387v2): a transformer policy learns sensor relocation improvement heuristics across climate-sensor instances | Learned relocation is an established approach. Our fresh per-map PPO currently does not provide this kind of amortized policy reuse. |
| Krause et al., JMLR 9(93), 2008 | [Paper and abstract](https://www.jmlr.org/papers/v9/krause08b.html): robust observation selection; section 6.5 includes sensor failures and structured simultaneous failures | N-1 or common-cause failures are useful modeling improvements, not independent novelty claims. |

This search did not establish that the proposed combination is new. The next
literature check should target the exact representation and algorithm after
they are specified, including adjacent camera/view-planning and decomposition
methods. No claim of exhaustive search or patent clearance is made.

## Priority 1: Make the Current Comparison Informative

1. **Audit active constraints.** Report endpoint slack, per-proposal violation
   counts, jointly violated constraints and feasible-start success rate. Identify
   structural redundancies separately from empirically loose constraints.
2. **Specify useful redundancy.** Use a prespecified critical N-1 threshold
   sweep or a Pareto study of coverage, cost and critical resilience. A possible
   alternative excess term is `sum_j max(d_j-r_j,0)`, where required multiplicity
   `r_j` is higher in critical regions. This is a proposed model change, not a
   reason to relabel the existing winner.
3. **Fix sampling semantics.** Choose either target quadrature weights with
   physical units or normalized coverage/importance/excess terms and an explicit
   cost scale. Test target-density sensitivity. Keep historical J unchanged.
4. **Audit accessibility.** Compute candidate/action connected components and
   anchor exclusions without raycasting. Compare local moves with feasible
   candidate relocation and coordinated two-sensor replacement. In an offline
   installation design, the sensor need not physically travel along the search
   trajectory. Final deployment constraints still apply.
5. **Strengthen baselines.** Compare same-domain multi-start/iterated LS and
   coordinated two-sensor search with MAPPO. Separate action-space ablations
   from optimizer comparisons. Include cold-start and small exact constrained
   references, actual evaluation/ray counts and equal-time results.

Do not launch the entire 720-pair grid to answer these questions. A diagnostic
slice should establish which constraints and search decisions affect behavior.

## Priority 2: Three Useful Application Extensions

### Spatial Coverage Gaps

An overall percentage does not describe the location or connected length of
uncovered regions. For each declared road route, partition its corridor into
longitudinal bins; evaluate each bin over a fixed width and target-height set.
Measure the longest consecutive stretch whose coverage is below a declared
threshold, including stretches across module boundaries.

This remains geometric coverage. It does not certify detection continuity or
tracking handover. Bin size, corridor definition, threshold and allowable gap
need application justification or clearly labeled sensitivity experiments.

### Service-Group Failure

Several sensors may share one power/data hub. Define the assignment as a design
variable and remove all sensors assigned to one hub in each failure scenario:

```text
R_hub(P,z) = min_h C(P excluding {i: z_ih = 1})
```

The current matching is only a feasibility witness; its arbitrary assignment
must not silently become the optimized fault-tolerance design. Jointly select
assignment and placement, or explicitly define a conservative assignment rule.
Shared conduits and upstream cabinets can require failure groups beyond hubs.
Structured failure modeling already exists in robust selection research
[Krause et al.](https://www.jmlr.org/papers/v9/krause08b.html).

### Site Cost and Geometric Uncertainty

Replace uniform height-only cost with existing-pole versus new-pole cost and
service-path connection cost. Use allowable route lengths rather than Euclidean
hub distance when site data become available. Initially use synthetic labeled
cost cases; do not present invented values as currency or engineering limits.

Separately evaluate mounting translation/rotation errors and changed obstacle
geometry with direct continuous-pose raycasting. Do not snap the perturbation
back to the nominal grid. Worst-case performance over a finite scenario list
certifies only that list, not every perturbation in a continuous uncertainty set.
These factors strengthen the application model but are not the central novelty.

## Priority 3: Transfer Responses and Boundary Information

### Library Contents

Store multiple source alternatives distinguished by branch/side allocation,
coverage/cost/resilience tradeoff and boundary observations. Avoid keeping only
the top few nearly identical solutions. Do not require one-to-one sensor
interpolation between incompatible allocations.

For a candidate pose, store or approximate its target-level visibility support,
geometry descriptor and validity conditions. Derive the current marginal value
from the other selected sensors:

```text
Delta(p | P_minus_i) = sum_j w_j * a_pj * 1[d_minus_i,j == 0]
```

A scalar response stored at one source deployment is conditional on that
deployment. It cannot be reused unchanged after other agents move. Target-level
support, or a model conditioned on the current deployment, is needed to update it.
Source and target target-sets can differ; correspondence/resampling error and
target weights must be explicit.

### Composition and the Earlier Sphere Idea

Keep the physical road and rays in their original Euclidean geometry. Give each
road module explicit connection ports and an exterior context band. Store the
candidate's outward visibility near each port, together with target identity,
range, orientation and occlusion information.

A sphere can be an optional direction-indexed descriptor around a module or
sensor, with radial layers and port labels. A direction-only image loses depth;
identifying opposite map edges invents physical adjacency. Test this encoding
against simpler port-relative coordinates before making it central.

At assembly, transform modules into one global frame, give targets unique
ownership, and evaluate visibility across actual connected geometry. Sum
per-target observations before applying coverage/redundancy tests. Pairwise
overlap subtraction alone is not exact when three or more sensors overlap.
Global sensor budget, hub capacities and costs still require joint allocation.

With finite sensing radius R, a target whose distance to its module boundary
is greater than R cannot be seen by a sensor outside that module. Its relevant
straight ray segments also remain inside the module. Thus, if interior geometry
and sensor models are unchanged, exterior-only changes cannot alter that
target's incidence values. This gives a useful locality condition for reuse.
It does not cover interior geometry changes or remove global budget coupling.
Context halos must contain all relevant ray segments and occluders.

This locality argument is elementary. Research value would come from a compact,
transferable interface representation and a demonstrated reduction in online
work while preserving joint feasibility and coverage.

### Selective Revalidation

Transfer responses as predictions, then prioritize real queries where uncertainty
could change feasibility or candidate ordering. Recompute geometry-sensitive
or cross-boundary rays; validate every final deployment independently.
Compare against uniform sampling, geometry-only risk ordering and a surrogate
trained without the source library. Surrogate sampling itself is not novel
[Chen et al.](https://www.sciencedirect.com/science/article/pii/S0968090X24003590).

Predicted confidence is not a certificate. A certifiable variant would maintain
binary incidence bounds `lower_ij <= a_ij <= upper_ij`, marking unverified
relations unknown unless a geometric argument establishes them. Their unions
give coverage bounds; lower/upper density bounds handle multiplicity thresholds.
N-1 checks retain sensor identity. Query cost for obtaining useful bounds must
be included, and intervals that stay too loose would falsify the practical idea.

## Mathematical Claims Worth Targeting

For a fixed finite true-feasible library family Q, suppose the surrogate
objective satisfies `|J(P)-J_hat(P)| <= epsilon` for every P in Q. Suppose the
returned P_hat maximizes J_hat over Q within additive eta. Then:

```text
max_{P in Q} J(P) - J(P_hat) <= 2*epsilon + eta.
```

Proof: at a true maximizer P_Q, add and subtract J_hat(P_Q) and J_hat(P_hat).
The two estimation errors contribute at most epsilon each; the surrogate
optimization gap contributes at most eta.

Against the full target feasible domain, add the library omission gap:

```text
J(P_full_star) - J(P_hat) <= delta_library + 2*epsilon + eta
delta_library = J(P_full_star) - max_{P in Q} J(P).
```

These are conditional bounds, not established guarantees of the current code.
Average prediction error does not establish a uniform epsilon. MAPPO does not
currently certify eta. Direct verification of only the returned deployment does
not bound delta_library. Restricting to a certified feasible subset may add to
that omission gap. The inequalities are standard error decompositions; a useful
method must obtain tight bounds or demonstrate low error without excessive work.

Do not apply the usual monotone coverage greedy guarantee to all new objectives.
For example, with one target and two identical sensors, worst-single-failure
coverage is zero with either sensor alone and one with both: the second sensor
has an increasing marginal contribution. Robustness can break submodularity.
Coverage lower bounds also do not define a downward-closed feasible family.

## Smallest Experiment That Tests the Main Idea

| Stage | Work | Decision evidence |
|---|---|---|
| A | Audit saved constraints and action accessibility; freeze objective semantics and comparison protocol | Which requirements are active, and whether accessible candidates explain failures |
| B | Compose two road modules in one geometry; create several source-only alternatives and a direct cross-boundary evaluator | Unique global targets, no artificial edge wrapping, agreement with direct raycasting |
| C | Compare pose-only transfer, response transfer, boundary-aware response transfer and cold-start search | Feasible success rate, endpoint and boundary quality, total online time and real ray counts |
| D | Apply LS, iterated/pair LS and MAPPO to the same candidate information and budgets | Whether learning contributes beyond representation and candidate accessibility |
| E | Evaluate on untouched geometry/constraint combinations and then larger module graphs | Generalization, scaling, library size and offline-cost amortization |

Use a small exhaustive/MILP reference on the SAME finite constrained domain.
Distinguish restricted-library optimum from a broader candidate-space reference.
Include all source selection, transfer, repair, query and verification costs
under a declared timing boundary. Keep software startup/export costs separately
visible. Publish failed or blocked cases, not only conditional quality.

Holdout changes should include independent variation in branch length, angle,
obstacles and approved poles, not only more seeds on the same width/offset maps.
New geometry types and assembled scenes require extending and validating the
shared physical engine before those experiments are available.

## Search Record

Sources searched through web queries: primary author manuscripts on arXiv,
JMLR, publisher pages including Elsevier and MDPI, CVF, Eurographics and author
repositories. Search cutoff: 2026-10-20; no lower year limit, so foundational
robust-selection work is included. English technical literature was searched.
This is not a Scopus/Web of Science systematic export.

Representative exact queries executed:

```text
roadside LiDAR placement optimization transfer learning geometry sensor placement
"LiDAR" "placement" "robust" optimization
"sensor placement" "transfer learning" coverage
"sensor placement" "boundary" "coverage" decomposition
"sensor placement" "generalization" "reinforcement learning"
"roadside" "LiDAR placement" "2025"
"sensor placement" "surrogate" "active learning"
"robust" "sensor placement" "submodular" Krause
"roadside lidar" "placement" "reinforcement"
"LiDAR placement" "2026" optimization
"Learning to Optimise Climate Sensor Placement using a Transformer"
"Roadside LiDAR Placement Optimization in Curved Tunnels"
"VF-Plan" "Visibility Field"
"Roadside LiDAR placement" "chance-constrained"
"sensor placement" "coverage" "spherical"
"sensor placement" "case-based"
```

The evidence table retains seven unique core works. Duplicate arXiv/publisher/
repository records were consolidated by identifier/title. This is not a count
of every retrieved hit. Secondary summaries, reviews and patent search hits
were not used as technical evidence. GBlobs (arXiv:2510.18539) was screened out
of the core comparison because it studies perception robustness to placement,
not a deployment optimization/transfer algorithm. Inaccessible full texts were
not inferred from titles; Chen 2024 and Ma 2026 are marked as publisher previews.
VF-Plan and Wang are identified by the preprint versions actually inspected.

## Audit Provenance

Read-only inputs, SHA-256:

```text
rsu_coverage_study.py
ebf9c8055cccf66399db030fdef858e0737b3d7027952f738c6a8219ed505bca
src/core/evaluation.py
b12fdac3cbc1cb758f95d26bc981382d9250bacffa593cd2b0f8ceb80613a32b
src/core/coverage_requirements.py
350cb318b9b9cb7af7d8418f68ed20eb9a50c251869492b6d62044233fc23954
src/methods/constrained_refinement.py
4211222ae92306e401bfa4c8c3c0a8965c88f7cb6e312c2aff4c34690b1e3545
prototype_segments.py
f00e9977f365c6444bfd637b4016853384897bf4033f0da415e37527585c3fec
results/rsu_coverage_development_20261020/raw_methods.csv
53c27d112b18a5491dfc3daee49abac9d095bf61444cbcf681de32e41864852c
```

Only this review document was added. Algorithm code, requirements, historical
results and README were not edited. Numerical checks use saved CSV/arrays;
they are not a new independent physical validation.
