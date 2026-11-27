# Roadside LiDAR Constraints and Wireless Reward Transfer

Review date: 2026-11-09, Asia/Seoul. Targeted scoping review, not a systematic
review, novelty clearance or regulatory compliance assessment. No experiments,
training, solver runs or implementation changes were performed for this review.
Detection and tracking remain deferred. All new model elements below are proposals.

## Recommendation

Keep the main question in [TITS_RESEARCH_PLAN.md](TITS_RESEARCH_PLAN.md): efficient
constrained placement on a road network with explicit remaining quality uncertainty.
Do not turn this into a collection of arbitrary reward terms or a requirement to
use MAPPO. Prioritize three additions to the experimental problem:

1. Route-level geometric continuity, especially through module connections.
2. Coverage under declared occlusion and installation-error scenarios.
3. Approved installation assets, actual power/backhaul connectivity, and shared costs.

Wireless research supplies useful objective structures, not a replacement for the
LiDAR visibility model. A V2X RSU is communication equipment; a roadside LiDAR is
a sensor that may share its site, power, backhaul and processing infrastructure.
Radio coverage, optical coverage and end-to-end availability are different quantities.

## Evidence Selected

Dates below are publication/version dates, not search-engine crawl dates. Primary
publisher, author and official infrastructure sources were used. Some publishers
blocked direct full-text access; those entries support only the stated abstract-level
claims, not uninspected equations or numerical thresholds.

| ID / source | Date, status and inspected evidence | Supported modeling motivation |
|---|---|---|
| S1. Zhang et al., [Multi-Factor Roadside Unit Deployment Optimization Considering Traffic Safety Risks](https://ieeexplore.ieee.org/document/10756282/) | T-ITS 26(2), 2252-2265, February 2025; online 2024-11-18. DOI 10.1109/TITS.2024.3491994. Publisher-indexed abstract/metadata. | Communication RSU deployment combines unequal vehicle demand, installation cost, geographic restrictions, road coverage and risk-informed priorities. It does not establish that geometric LiDAR coverage reduces crashes. |
| S2. Ren et al., [Road Side Unit Location Optimization Considering Communication Channel Competition and 6G Technology](https://portal.fis.tum.de/en/publications/road-side-unit-location-optimization-considering-communication-ch/) | T-ITS 26(7), 9867-9881, 2025. DOI 10.1109/TITS.2025.3572183. Author-institution abstract/metadata. | A budgeted stochastic placement/allocation model considers demand uncertainty, communication delay, packet loss/collisions and age of information. Coverage alone does not represent communication service. The modeled 6G setting is not evidence of a deployed 6G system. |
| S3. Anzola-Rojas et al., [Optimizing road side units connectivity in intelligent transportation systems with optical network solutions](https://www.sciencedirect.com/science/article/pii/S2214209625001020) | Vehicular Communications 56, 100975, December 2025. DOI 10.1016/j.vehcom.2025.100975. Publisher-indexed abstract and [author-project abstract](https://onofre4.tel.uva.es/publication/anzolarojas-2025100975/); full PDF blocked. | Optical backhaul planning includes civil works, cable and device CAPEX; PtP/PON alternatives are modeled with ILP and a genetic algorithm. This motivates shared connection costs rather than only a per-sensor price. |
| S4. Ma et al., [Roadside LiDAR placement optimization in curved tunnels: A framework integrating probabilistic grids and neural radiance models](https://www.sciencedirect.com/science/article/pii/S0886779826001872) | Tunnelling and Underground Space Technology 173, 107629, July 2026. DOI 10.1016/j.tust.2026.107629. Publisher-indexed abstract/introduction. | Curvature-related static occlusion and traffic-related dynamic occlusion affect placement; both fixed-sensor and minimum-sensor problems are studied. Curved geometry, probabilistic grids and a learned surrogate are already related work, not novelty by themselves. |
| S5. Chen, Zheng and Tan, [Roadside LiDAR placement for cooperative traffic detection by a novel chance constrained stochastic simulation optimization approach](https://www.sciencedirect.com/science/article/pii/S0968090X24003590) | TR-C 167, 104838, October 2024. DOI 10.1016/j.trc.2024.104838. Publisher-indexed abstract/introduction inspected in the existing research-plan review; full text not obtained. | Traffic uncertainty, sensor budget and a recall chance constraint motivate scenario-aware placement. Their detection objective is outside our current scope; do not relabel geometric visibility as recall. |
| S6. SNS JU, [5G Strategic Deployment Agenda for CAM, Road Chapter](https://smart-networks.europa.eu/wp-content/uploads/2026/01/sns_5g_sda_for_cam_on_road.pdf) | Approved 2025-12-18; published 2026-01-08. Official strategy document, not a LiDAR installation standard. PDF sections 6.2-6.3 and 7.3 inspected. | Site space/structural capacity, power, access to land/fiber, passive infrastructure sharing and backhaul alternatives matter. This supports asset-based feasibility inputs, not universal mounting or redundancy thresholds. |
| S7. Toth, Perry and Keshmiri, [Connected and Automated Vehicle Readiness Action Plan](https://rosap.ntl.bts.gov/view/dot/82913/dot_82913_DS1.pdf) | TDOT/FHWA-sponsored report RES2023-29, 2025-03-28. PDF printed pp. 21-25 and 41 inspected. | Existing power/communications, infrastructure upgrades, cabinets/poles/switches and ongoing operations/maintenance affect deployment. Its US radio-RSU cost estimates are not LiDAR prices or Korean construction budgets. |
| S8. Kulkarni et al., [Maestro: QoE-Aware Dynamic Resource Allocation in Wi-Fi Networks](https://doi.org/10.1145/3709371) | Proceedings of the ACM on Networking 3(CoNEXT1), article 4, 2025-03-06. Publisher-indexed abstract; [NSF-hosted indexed reward excerpt](https://par.nsf.gov/servlets/purl/10607921). Full PDF retrieval failed. | Wi-Fi resource allocation explicitly considers overall service quality and fairness, rather than aggregate throughput alone. This is scheduling/resource allocation, not AP placement or a LiDAR reward validation. |
| S9. Nunez et al., [Deep Reinforcement Learning-Based Scheduling for Wi-Fi Multi-Access Point Coordination](https://ieeexplore.ieee.org/document/11478468/) | IEEE TMLCN 4, 744-757, published 2026-04-09. DOI 10.1109/TMLCN.2026.3682239. Publisher metadata plus [2025 author full-text version](https://arxiv.org/html/2507.19377v1), section IV-C. | PPO scheduling targets worst-case delay, masks invalid/unproductive actions and combines immediate improvement with delay-dependent reward. Borrow the bottleneck-service and feasibility concepts, not packet-delay formulas for static visibility. |
| S10. [Robust Relay Placement for Millimeter-Wave Backhaul via Link-Quality-Based Cooperative Coevolution](https://ieeexplore.ieee.org/abstract/document/11371621) | IEEE Wireless Communications Letters 15, 1742-1746, published 2026-02-04. DOI 10.1109/LWC.2026.3661163. Publisher-indexed abstract/introduction. | Relay placement couples blockage, routing, interference and graph reliability; the objective considers outage/worst-path success. It uses evolutionary optimization, not RL: the transferable part is the network objective and constraints. |
| S11. Wu et al., [Intelligent Optimization of Wireless Access Point Deployment for Communication-Based Train Control Systems Using Deep Reinforcement Learning](https://arxiv.org/html/2509.24819v1) | arXiv v1, 2025-09-29; journal acceptance not verified. Author full-text sections II-B and III-B inspected. | A fixed-three-AP example minimizes a mixture of mean and worst penalized path loss; reward is negative next-state cost, equation (20). Here 'cost' is a signal-quality loss, not an independently verified installation CAPEX term, despite broader abstract wording. |
| S12. Cui et al., [MR-LiDAR: A Multi-Resolution Roadside LiDAR Benchmark for Perception Diagnostics and Deployment Guidance](https://arxiv.org/abs/2605.24777v2) | Author abstract, v2 dated 2026-10-14; v1 2026-05-23. Related IOP DOI is listed, but journal publication status was not independently verified. | Beam layout, range, mounting height and curved-road conditions affect point acquisition. Keep this as motivation for a later sensor-resolution model; binary ray visibility is not point density or detection quality. |

## Current Code Versus Missing Modeling

Checked against [CoverageRequirements](src/core/coverage_requirements.py),
[constrained refinement](src/methods/constrained_refinement.py),
[RSU protocol](RSU_COVERAGE_PROTOCOL.md),
[corridor config](configs/connected_corridor_pilot.json) and
[geometry profiles](configs/geometry_placement.json).

| Topic | Already available | Missing or oversimplified |
|---|---|---|
| Installation | Pose grid, allowed/forbidden regions, height window, spacing and identical-XY pole capacity | Surveyed pole/site IDs, bracket orientation limits, approved structural/clearance/access data |
| Budget | K bound; base/height/region sensor cost and total cost cap | Variable-count refinement, shared site activation, trench/cable route cost, lifecycle costs |
| Coverage | Global/ROI union, importance, lower/upper double-coverage fraction, excess-observation penalty | Lane/route gap lengths, route ordering across junctions, critical co-observation interval lengths |
| Infrastructure | Matching sensors to combined power/data hubs by horizontal reach and port count | Separate electrical/data networks, routed cable length, watts, bandwidth, processing demand and shared failure domains |
| Robustness | Any-one-selected-sensor failure coverage | Common cabinet/fiber/power failure, declared traffic occlusions and pose uncertainties |
| Physics | Shared static mesh, range/FOV and per-sensor visibility masks | Validated beam sampling, weather attenuation and downstream perception |

Availability in the framework does not mean every feature was enabled in the
latest experiment. The latest mechanism study uses fixed K=2, the corridor cost,
spacing/height/coverage/N-1 settings, and base/combined connection profiles. It is
not a test of surveyed service hubs, a citywide network or all constraints above.
Existing numeric thresholds are illustrative; the user has not specified operational
budget, clearance, overlap or service requirements.

## Selected Additions

The following are our proposed adaptations, not requirements quoted from the papers.

### A. Route Service and Useful Overlap: First Priority

For each declared lane/path, retain ordered arc-length samples and globally unique
target IDs. Define the sample-height/width aggregation rule before evaluation.
Measure route coverage and the longest consecutive unobserved interval in meters.
Use arc length, not straight-line endpoint distance through a curve. Preserve a
gap that crosses a module boundary; resetting the statistic at every module hides it.

Add minimum coverage for specified approaches/critical zones and a maximum gap
where required. A 95% global score can still leave one long segment uncovered.
This is motivated by bottleneck service in S9/S11 and extends the existing ROI
screen, but it is a geometric proxy, not an object-tracking handover guarantee.
Any conversion to time requires an explicit speed profile and frame-rate assumptions.

Retain overlap only where it has a declared role: critical co-observation regions
or backup coverage. Count distinct physical sensors/sites; optionally require
distinct power/backhaul failure domains. View-angle diversity can be a later
geometric metric, but no universal angular threshold is justified here. Overlap
is not radio interference and is neither always beneficial nor always wasteful.
Do not copy Wi-Fi roaming overlap percentages into LiDAR requirements.

### B. Scenario Robustness: First Priority

Construct explicit mesh/pose scenarios: empty road, queued large vehicles, a bus
blocking a corner, temporary roadside works, or bounded mounting/yaw/pitch errors.
These use geometric raycasting without a detector. For every scenario, define valid
target locations and weights consistently: targets inside a solid occluder must not
be silently counted as observable free-space service points. Keep the same labeled
service routes/ROIs, and disclose any changes in eligible target sets/denominators.

Report nominal, per-scenario and worst declared-scenario coverage/gap metrics.
With no calibrated occurrence probabilities, use a finite robust scenario set,
not an invented '99.9% reliability' claim. Expected loss, CVaR or chance constraints
are later options only with a stated probability model and held-out validation.
Sampled stress tests do not certify every possible traffic condition. S4/S5 motivate
uncertainty; our detector-free robust service definition is a separate choice.

Installation errors and traffic variation are physical uncertainty. Unqueried
visibility in the selective-evaluation method is computational uncertainty.
Keep them separate; a learned confidence score is not an exact incidence bound.

### C. Infrastructure Graph and Shared Cost: First Priority

Use approved site IDs with allowed poses, cabinet/power connections and permitted
cable/trench routes. Separate three structures:

- Road graph: route order, branches and connection regions.
- Optical visibility relation: sensors to targets, including nearby roads not adjacent in the road graph.
- Infrastructure graph: power/data paths, capacities, shared assets and failure domains.

A sensor counts as service-available only if its declared power and data service
conditions hold. Infrastructure availability may be binary in the first model;
do not turn port counts into an end-to-end latency guarantee.

Represent total cost as sensor cost + activated-site cost + activated-link cost
+ declared maintenance cost over a fixed horizon. Charge shared cabinets/trenches
once, and distinguish pre-existing assets from new investment. Use actual routed
lengths and connector/elevation allowances rather than horizontal reach alone.
S3/S6/S7 support these cost categories, not prices for our site.

Add aggregate electrical demand and declared raw/compressed data-rate bounds at
hubs/links. Capacity needs depend on the selected sensor and data product, not just
how many road targets are visible. Initially use fixed engineering demands; actual
queueing delay, scheduling, packet loss and age of information require a separate
communication model. Apply such a model only if communication co-design becomes
part of the research question; S2 is not evidence that we already model it.

### D. Failure Domains and Physical Fidelity: Second Priority

Extend sensor N-1 to a specified cabinet, shared fiber segment or power-feed outage.
Two sensors with one common failed hub are not two independent backups. Define
surviving service paths and recompute available-sensor coverage; do not assume
independent probabilities or multiply link reliabilities without evidence. S10
motivates graph reliability, while the failure-domain selection is our design choice.

Add surveyed grades, overpasses, buildings and vegetation before labeling maps
realistic. Use manufacturer-supported sensor modes and beam layouts if adding
point acquisition (S12). Mounting approvals, clearance, access for maintenance,
eye safety and environmental suitability should be external eligibility inputs
checked by the responsible owner/engineer, not adjustable reward penalties.
This review did not establish jurisdiction-specific legal requirements.

Defer detection/tracking/fusion metrics, full packet-level radio simulation,
weather-dependent optical physics and maintenance scheduling. They need additional
validated models and would obscure the present placement/query-efficiency question.

## Borrowing Rewards From Wi-Fi and Relays

| Borrowed structure | LiDAR adaptation proposed here | Boundary |
|---|---|---|
| Mean and worst-location quality, S11 | Mean route coverage plus worst route deficit/longest gap | Replace RF loss with geometric service measures; do not use dBm, tunnel multipath or the paper's numeric penalty coefficients |
| Service fairness, S8 | Per-approach service floors; optional worst normalized deficit | Do not maximize fairness alone: uniformly poor coverage can be perfectly equal |
| Worst-delay reduction and action masks, S9 | Reward reduction of the largest service deficit; reject installation/infrastructure-infeasible joint layouts | Scheduling a queue is not static placement; per-agent masks alone cannot enforce all joint constraints |
| Outage and graph reliability, S10 | Service coverage after a specified shared-infrastructure failure | No automatic independence, radio success probability or optical-reflection model |
| Network CAPEX, S3 | Marginal site/link activation cost, with shared infrastructure charged once | This is an optimization cost term, not originally an RL reward; all baselines can use it |

Do not copy SINR, co-channel interference, contention/backoff, RF transmit power
or packet-delay rewards into binary LiDAR coverage. Optical sensor interference
would require its own measured physical model. Separate a radio subsystem if it
is actually being designed. Also do not import aircraft propulsion/collision costs
from aerial relay papers into permanently mounted roadside sensors.

### Proposed Common Optimization Contract

For selected candidates x, activated sites y and infrastructure links z, let
a[i,j,w] be physical visibility under declared scenario w and b[i,w] indicate that
the selected sensor has functioning power/data service in that scenario. Define:

```text
d[j,w] = sum_i x[i] * b[i,w] * a[i,j,w]
C[r,w] = weighted fraction of route/ROI r with d[j,w] >= 1
G[r,w] = longest unobserved arc-length interval on route r
B = sensor costs(x) + site activation costs(y) + link costs(z) + declared O&M
```

An initial formulation is maximum weighted geometric coverage under B <= B_max,
approved installation/capacity conditions, C[r,w] >= tau[r,w], G[r,w] <= g[r,w],
and any declared critical redundancy requirements for all modeled scenarios.
Nominal and contingency thresholds may differ, but must be fixed in advance.
Use minimum cost subject to service instead if the actual planning decision is
the required number of sensors. Neither formulation is implemented by this memo.

For optional soft tradeoffs, define normalized route deficits and utility:

```text
D[r,w] = max(0, (tau[r,w] - C[r,w]) / tau[r,w])   # only for tau > 0
J = mean_w weighted_coverage(w)
    - lambda_D * max_(r,w) D[r,w]
    - lambda_G * max_(r,w) (G[r,w] / route_length[r])
    - lambda_B * B / B_scale
```

Uniform mean_w is a declared design weighting, not a measured expectation.
Under hard coverage floors, D is zero on feasible solutions and can be used for
repair diagnostics rather than advertised as an additional preference. Choose
hard requirements versus soft preferences deliberately. Normalize metrics, freeze
weights on development data, and report each component/Pareto tradeoff separately.
Do not impose arbitrary numerical service thresholds just because a paper used them.

If MAPPO is later compared, use the same J and feasible domain as LS/exact methods.
A shared improvement reward r[t] = J(P[t+1]) - J(P[t]) is a reasonable starting
point, not a new contribution. Its undiscounted finite-horizon sum telescopes to
the final improvement; discounting, reward clipping or other shaping changes
that equivalence. Preserve the historical reward/configuration in separate runs.
Required installation/service conditions must be checked explicitly; a large
penalty does not ensure they hold. [CPO (ICML 2017)](https://arxiv.org/abs/1705.10528)
is relevant background on constrained RL, not proof that our MAPPO enforces
per-layout constraints or is globally optimal.

Agent credit D_i = J(P) - J(P_without_i) can be investigated only after defining
the counterfactual. Removing a sensor can violate a service floor; evaluate a
well-defined soft utility or use an explicitly feasible replacement baseline,
not an undefined feasible-only score. Simultaneous agent moves still need joint
evaluation. Fixed K makes a pure sensor-count penalty constant; it cannot teach
sensor removal without an appropriate action space.

Keep placement reward separate from selective-query control. A queried pose is
not necessarily installed. The latter problem trades evaluation work against
verified final feasibility/quality; it must not assume unseen visibility while
scoring actions. Reuse any learning only after testing cross-map generalization
and counting training costs. Extra constraints do not establish a MAPPO advantage.

## Bounded Next Work

1. Write a service-contract input with units, provenance and status for every
   threshold: measured, owner-specified, manufacturer-specified or illustrative.
   Required assets: 3D mesh/coordinate frame, lane routes, approved sites/poses,
   power/data graphs, device demand, costs, critical regions and stress scenarios.
2. Add route gap evaluation and tiny boundary-crossing counterexamples first.
   This reuses current masks and directly tests the road-connection motivation.
   Audit nonuniform sampling, route direction, sample aggregation and resolution.
3. Add a small, separate scenario layer and fixed infrastructure graph. Validate
   one-long-gap versus scattered holes, one shared versus two independent hubs,
   cable-route versus Euclidean distance, and multi-sensor loss from one outage.
   Check impossible contracts explicitly; a search failure is not infeasibility.
4. Continue Gate 2 tiny exact-bound tests and Gate 3 global allocation, extending
   the proof obligations only for constraints actually enabled. Optional overlap
   penalties are nonmonotone; upper visibility substitution is not automatically
   an upper objective bound. Do not inherit a pure-union greedy guarantee.
5. Before larger runs, freeze new untouched geometry families and compare full
   visibility/exact small references, static-pair, no-feedback, constructive/repair
   heuristics and coordinated or restart LS. MAPPO remains optional. Report feasible
   rate, per-route/worst-scenario deficits, max gap, cost, rays and end-to-end time.
   Separate each constraint family and its combination; keep infeasible cases.

The latest [mechanism controls](GEOMETRY_MECHANISM_STUDY.md) do not demonstrate
a measured-feedback or wall-time advantage. Do not add constraints to manufacture
a win. A defensible contribution would be a validated reduction in evaluation
work/time for this coupled service problem, with useful finite-domain bounds,
not the presence of known RSU constraints or a wireless-inspired reward alone.

## Search and Access Record

Main search window: 2024 through 2026-11-09, with older foundational work only
where needed. Inclusion: primary placement/optimization studies, relevant wireless
reward methods, and official infrastructure reports. Exclusion from core evidence:
detector-only/tracking-only work, surveys as proof of a technical result, generic
AI-generated summaries, and a 2025 Technical Disclosure Commons AP/RL disclosure
as if it were a peer-reviewed paper. SSRN and the published curved-tunnel paper
were treated as one work, not two independent confirmations.

Representative queries actually used:

```text
roadside LiDAR placement optimization constraints occlusion redundancy 2025 2026
roadside unit deployment optimization infrastructure power backhaul constraints 2025 2026
Wi-Fi access point placement optimization 2025 reinforcement learning reward coverage fairness backhaul
relay placement optimization 2024 2025 reinforcement learning reward connectivity energy outage backhaul
"Roadside LiDAR Placement Optimization in Curved Tunnels"
"Robust Relay Placement for Millimeter-Wave Backhaul"
"Maestro" "QoE" "fairness" "reward" Kulkarni
roadside unit placement optimization 2025 power backhaul reliability network coverage constraints
"Multi-Factor Roadside Unit Deployment Optimization Considering Traffic Safety Risks" IEEE
"Road Side Unit Location Optimization Considering Communication Channel Competition" IEEE
"Optimizing road side units connectivity" "100975"
"Deep Reinforcement Learning-Based Scheduling for Wi-Fi" "2026" DOI
```

Search results were followed to publishers, author institutions, arXiv and
official reports. The EU PDF was also downloaded and text-extracted to inspect
the infrastructure sections after browser timeouts. This was not an exhaustive
database export or a replication of the cited experiments. No universal overlap,
mounting separation, service reliability or installation budget was established.
