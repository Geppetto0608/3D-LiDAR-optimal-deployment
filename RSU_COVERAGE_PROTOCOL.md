# RSU Coverage Study Protocol

This protocol concerns **static geometric coverage of a roadside LiDAR installation**.
A LiDAR is the sensing part of an infrastructure system, not the V2X radio itself.
Detection, tracking, communication reliability and road-safety performance are not
inferred from geometric visibility.

For the 2026-11-09 literature-based extension priorities, see
[RSU infrastructure and wireless reward review](RSU_INFRASTRUCTURE_REVIEW_20261109.md).
Route gaps, scenario robustness and graph-based infrastructure/shared costs there
are proposed extensions, not additions to the implemented schema below. Existing
thresholds and historical experiments are unchanged.

## Evidence and Scope

| Primary source | Supported motivation | What this study does not borrow or claim |
|---|---|---|
| Vijay et al., ITSC 2021, [Optimal Placement of Roadside Infrastructure Sensors](https://arxiv.org/html/2110.01251v1) | Raycast visibility, cost and redundancy matter for infrastructure sensor placement; road features and obstacles belong in the environment model. | Our thresholds, transfer method and MAPPO advantage do not follow from this paper. |
| Kim et al., Sensors 2023, [Placement Method of Multiple Lidars](https://www.mdpi.com/1424-8220/23/21/8808) | Urban LiDAR placement has restricted installation areas and blind spots; overlapping observations can have a useful role. | Target visibility counts here are not the paper's beam/point-density model or detection accuracy. |
| Du et al., IET ITS 2022, [Quantifying Performance and Optimizing Placement](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/itr2.12185) | Cost and coverage constraints are relevant, and placement performance depends on traffic/occlusion assumptions. | Its traffic-dependent perception model and numerical accuracy requirements are not implemented here. |
| Sensors 2016, [A Fault Tolerance Mechanism for On-Road Sensor Networks](https://www.mdpi.com/1424-8220/16/12/2059) | Failure and backup deployment create a cost/resilience tradeoff in road sensor networks. | This is adjacent network research, not a prescribed LiDAR N-1 coverage threshold. Our leave-one-out coverage constraint is an engineering modeling choice. |
| Ouster, [Sensor Quick Start](https://docs.ouster.com/sensor-docs/overview/quick-start-guide) and [Hardware Accessories](https://docs.ouster.com/sensor-docs/hardware/rev7/accessories) | Real devices require power and a data interface; cabling has physical limitations. | These interfaces motivate a service-reachability screen. They do not specify the Velodyne-like model's electronics, an actual cable route, throughput or a universal distance limit. |

Accessed 2026-10-20. Sources support **why a factor belongs in a model**, not the
illustrative thresholds below. No authority has specified the site's required
coverage, overlap, budget or mounting clearance. Those remain unknown.

## Implemented Requirements

The new schema is `constrained_coverage_v2`. Existing v1 inputs, physics, default
CLIs and historical outputs retain their behavior. Both LS and MAPPO use the
same `CoverageRequirements` evaluator and feasible-deployment objective J.

| Requirement | Evaluation | Status of numerical values |
|---|---|---|
| Installation geometry and sensor pose | Shared installable cells, XYZ/yaw/pitch grid, range/FOV/static mesh occlusion | Existing parametric model, not a surveyed site |
| Sensor budget | Fixed selected count during refinement, at most K in validation | K=3,4 experimental budgets; not minimum-sensor optimization |
| Installation cost | Base + absolute mounting height cost, bounded total | Arbitrary cost units, not currency or lifecycle estimates |
| Pole capacity and spacing | Maximum sensors at identical XY; minimum pairwise XY distance | Research exclusions, not regulatory clearance |
| Forbidden/approved installation areas | Rectangle blacklist and whitelist intersected with shared physical domain | Illustrative approved zones, not real pole coordinates |
| Mounting-height window | Additional inclusive height range inside physical Z grid | Example 3-5m, not a universal roadside standard |
| Power/data service reachability | Capacitated bipartite assignment from sensors to declared combined service hubs | Example 35m horizontal reach, 2 ports per hub |
| Global union coverage | Covered target fraction | Example minimum 75% |
| Per-approach coverage | Separate west/east/south/north target ROIs | Example minimum 60%, prevents ignoring one branch |
| Critical-region coverage and redundancy | Junction coverage and fraction observed by at least 2 sensors | Example minimum 80% and 15%; not tracking continuity |
| Excess observation tradeoff | Penalize sum(max(d_j-1,0)); separately bound double-covered target fraction | J penalty .05; global ratio upper bound .9 |
| Any-one-sensor failure | Minimum surviving global and critical-region coverage over each selected sensor's removal | Example 50% globally, 40% in junction; no failure probability assumption |

Service hubs are assumed to supply both power and data. A feasible assignment
is only a **horizontal reachability and port-capacity screen**. Cable paths,
vertical cable length, voltage drop, compute capacity, radio propagation,
latency and shared-hub failure are not modeled. A real deployment must replace
this screen with surveyed infrastructure and validated engineering constraints.

The critical-region overlap constraint is a necessary geometric proxy for
multi-view redundancy, not sufficient for fusion, association or handover.
The global excess-overlap penalty must not erase required critical redundancy;
hard thresholds take precedence over J.

## Failure Coverage Definition

For sensor masks a_ij and density d_j = sum_i a_ij, after sensor i fails:

```text
C_minus_i = count_j(d_j - a_ij >= 1) / N
C_N1 = min_i C_minus_i
     = (count_j(d_j >= 1) - max_i count_j(d_j == 1 and a_ij == 1)) / N
```

Apply the same formula to targets inside each critical ROI. Density alone does
not identify which sensor owns uniquely covered targets, so enabled failure
constraints require per-sensor masks and fail closed if they are absent.
Workers independently recast initial, best and final-policy deployments and
save those masks. This finite-set identity is exact for the chosen static
visibility model, not a reliability guarantee under weather or correlated faults.

## Learning and Comparison

The 128-step/one-update run was a wiring and constraint-sensitivity pilot, not
adequate evidence about MAPPO's performance. The expanded ladder uses fresh
policies with requested training counts:

| Joint steps | PPO updates at rollout=256 | Refinement soft cap |
|---:|---:|---:|
| 4,096 | 16 | 600s |
| 20,000 | 79, including residual rollout | 2,400s |
| 100,000 | 391, including residual rollout | 12,000s |

These are study budgets, **not convergence criteria**. If the time cap prevents
reaching the step count, `training_budget_completed=false`; do not label such a
run as a completed 100,000-step experiment. Record actual PPO diagnostics,
rejection fraction, final deterministic policy and best-seen deployment separately.
An improvement in best-seen J is not proof of a converged policy.

For each job the source is fixed by topology, using the low-width prototype.
No target coverage is used to select between sources. Both methods start from
one common feasible repair, retain original transfer anchors and have the same
radius and refinement time cap. The repair is deterministic, common preprocessing,
not an extra LS optimization stage granted to one method. Failure to repair is
not a proof of global infeasibility.

Report equal-time J using actual discovery timestamps, plus final J, total
coverage, ROI metrics, worst single-failure coverage, violations, cost, rays,
actual steps/updates and wall time. A method that terminates early retains its
already discovered best at later deadlines. Do not treat late final evaluation
improvements as if they were available before a deadline.

Strict best-improvement LS remains the current comparator. Random search,
random-restart/iterated LS, coordinated pair moves and an exact small-domain
constrained reference are still required stronger baselines. More constraints
do not imply MAPPO superiority; hard rejection can also prevent exploration.

## Maps and Execution

`configs/rsu_coverage_study.json` includes the four prior development targets
420001-420004 and four additional synthetic width/junction-offset maps
430001-430004, all served by the existing four-source library. They cover two
topologies, not eight independent real roads. All eight are **development** maps.
Curvature, skew intersections, slopes, custom obstacles and connected citywide
composition are not supported by this study.

The full configured grid is 8 targets x 2 K x 3 requirement profiles x 5 seeds
x 3 training budgets = **720 paired jobs / 1,440 method runs**, not yet executed.
Profiles are coverage/ROI/cost, plus installation infrastructure, plus N-1
resilience. This separates which family of requirements causes a change.
Default invocation only prints a plan; it does not read source solutions,
write files, raycast, repair or train.

This is an available factorial grid, not a recommendation to launch everything
at once. Its sum of configured refinement soft caps is 2,000 CPU-hours before
repair/import/validation; this is neither a runtime forecast nor a hard bound.
First choose a learning budget on a small development slice, then fix that
budget for the map/constraint study and a later untouched evaluation set.

```bash
.venv/bin/python rsu_coverage_study.py

# Bounded development check, not the full study.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u rsu_coverage_study.py \
  --target-ids 420002 420004 --k-values 4 --profiles resilient \
  --seeds 0 --step-budgets 4096 \
  --output results/rsu_coverage_development_20261020 --execute

# Longer learning slice, to run separately with a NEW output directory.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u rsu_coverage_study.py \
  --target-ids 420002 420004 --k-values 4 --profiles resilient \
  --seeds 0 1 2 3 4 --step-budgets 20000 \
  --output results/rsu_coverage_long_learning --execute

# Recheck saved arrays and regenerate derived CSV/Markdown only.
.venv/bin/python rsu_coverage_study.py \
  --output results/rsu_coverage_development_20261020 --report-only

PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_rsu_coverage test_constrained_coverage \
  test_feasibility_repair test_constrained_sensitivity test_pipeline
```

Sources, selected prototype/config/geometry inputs and requirements are frozen
and hashed before execution. Workers run sequentially in fresh CPU processes;
target exhaustive caches and pretrained policies are not used. Intermediate
stochastic traces are not independently recast point by point. Reports verify
saved masks without new raycasts. Output roots cannot be overwritten and there
is no optimizer/rollout resume. Generated `results/` are ignored by git.

## Completed Development Check

The bounded command above completed on 2026-10-20, 08:30:48-08:35:32 UTC.
Both requested MAPPO jobs completed 4,096 training steps and 16 PPO updates;
neither was truncated. Both LS jobs and all saved method deployments passed
worker direct-raycast validation and the reporter's per-sensor-mask checks.
There was one common repair on the T junction; the cross placement was already
feasible. The four runs use K=4 and seed=0, not the old K=3/128-step conditions.

| Target | LS coverage % | MAPPO coverage % | LS worst single-failure coverage % | MAPPO worst single-failure coverage % |
|---|---:|---:|---:|---:|
| 420002, T junction | 99.903 | 99.673 | 67.603 | 67.833 |
| 420004, cross | 99.836 | 98.990 | 75.873 | 76.258 |

The MAPPO-minus-LS objective differences are -268.0 and -491.35, respectively.
The small improvements in single-failure coverage are not improvements in J,
which is the prespecified selection criterion. This single-seed check is not
evidence of robustness superiority or a final judgment about MAPPO.
The 20,000/100,000-step experiments and the full grid remain unexecuted.
Raw data: `results/rsu_coverage_development_20261020/raw_methods.csv`, paired
differences: `paired.csv`, provenance: `manifest.json` and
`validation_summary.json`. Experiment code is frozen under `frozen/`; the
later enhanced saved-array reporter is identified separately in `report_source/`.

## Evidence Still Needed for a Paper

1. A real application owner must supply site pole positions, mount limits,
   budgets, service connections and critical road regions. Label arbitrary
   thresholds as sensitivity choices until then; do not cite them as standards.
2. Use development maps to select a fixed protocol, then create untouched
   evaluation geometries/sites. Do not repeatedly tune and report the same maps
   as unseen generalization. Five algorithm seeds do not replace independent maps.
3. Compare transfer-only, transfer+repair, transfer+LS, transfer+MAPPO and
   cold-start constrained optimization. Add stronger stochastic/local baselines
   with equal wall time; include repair and all common preprocessing once.
4. Evaluate threshold sweeps and incremental constraint profiles, not just one
   favorable difficult instance. Report infeasible/blocked rates and conditional
   quality together, with paired uncertainty across maps.
5. Fit training budgets from learning curves and diagnostics, not from which
   method wins. Neither 20,000 nor 100,000 steps alone establishes convergence.
6. Demonstrate geometry-aware transfer preserves or recovers constrained quality
   with fewer target-map queries/costs. Adding known constraints or MAPPO is
   not itself the claimed novelty, and no near-optimal bound is established here.
