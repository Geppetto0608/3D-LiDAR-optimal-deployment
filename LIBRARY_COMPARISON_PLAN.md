# Fixed A/B Library Comparison Before Expansion

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan, recording the agreed comparison-first decision
- Origin Date: 2026-11-19, Asia/Seoul
- Verification Status: UNVERIFIED for new experiments; no new run launched
- Version Label: library_comparison_plan_v1

## Decision

Freeze the current non-learned A/B methods and compare them against strong
source-free controls before expanding the library, obstacles, costs, operational
constraints or RL. The question is whether reusable deployment information adds
value under a controlled evaluation budget, not whether A beats B.

Track A transfers whole-topology deployments. Track B transfers components and
jointly selects poses. Analyze each track separately. The current B implementation
reuses one corridor component twice; it is not an arbitrary road-network solver.
TR-C is the current development target, not a claim of novelty or publication
readiness. The older T-ITS plan and negative experimental findings remain intact.

This document records the decision **after** the November 18 results. It is not
a retrospective preregistration. The original execution protocol remains
[LIBRARY_QUERY_CONTROLS.md](LIBRARY_QUERY_CONTROLS.md), unchanged.

## Frozen Reference

Use the completed development comparison as the reference, without rerunning it
merely to obtain another favorable outcome. Results and source snapshots remain
under their existing roots; do not overwrite them.

| Setting | Fixed reference |
|---|---|
| A source | `results/library_ab_smoke_20261118/a_v2` |
| B source | `results/library_ab_smoke_20261118/b` |
| Source layouts | Existing K1/2/3 frontiers, up to 3 alternatives per K including the incumbent |
| Target specifications | `configs/library_target_a_w9.json`, `library_target_a_w12.json`, `library_target_b_turn45.json`, `library_target_b_turn60.json`, all under `configs/` |
| Sensor and physics | Frozen source sensor/config, existing range/FOV and mesh LOS evaluator |
| Candidate and target grids | Existing per-track grids; identical across methods within a case |
| Objective | Unweighted target union coverage, at-most-K, K = 1, 2, 3 |
| Shared pose-query budget q | A9: 11; A12: 8; B45: 20; B60: 20 |
| Pool selection | Same exact subset enumeration and tie rules for every method |
| Controls | `geometry_single`, `geometry_union`, `geometry_portfolio`, `random` seeds 0..29 |
| Quality reference | Independent full-candidate enumeration, evaluated after all arm decisions |
| Refinement and new constraints | None; no LS/MAPPO, learned retrieval, cost or redundancy objective added |

Source masks and optimality certificates apply to their original scenes only.
Transfer candidates must be evaluated on the target mesh. Existing physical
installation rules remain active; "no new constraints" does not allow invalid poses.
Do not silently densify the grids or add candidate poses from target optima.

Completed artifacts and the exact reproduction commands are linked from
[README.md](README.md). The reference is
`results/library_query_controls_20261118/`: four development maps, 136 map/arm
evaluations, 408 K rows and 12 full-reference rows. These are not independent
holdout maps. The original per-run input/source hashes identify the execution,
not the current checkout alone. The ignored result directories are not included
in a fresh clone.

## Comparison And Interpretation

For target-map visibility sets A_i and candidate pool Q_m of method m:

```text
f(S) = |union(i in S) A_i|
L_m(K) = max f(S), S subset Q_m, |S| <= K
shortfall_m(K) = f*_full(K) - L_m(K)
delta_m_pp(K) = 100 * (L_library(K) - L_m(K)) / N_targets
```

The full optimum above is certified only in the small declared candidate domain.
If a future reference is time-limited, report its incumbent and valid bound
separately; an incumbent difference is not optimality regret.

Report every map/K and every predeclared control, including losses and failed
transfers. Primary comparisons are against both geometry union and portfolio;
random and single ranking remain diagnostic controls. Do not choose the weaker
control afterward or present the per-case best geometry method as an available
online selector without accounting for that selection.

Record covered targets, shortfall, actual sensor count, distinct pose queries,
submitted rays, optical prediction work and separate timing components. Equal q
does not equalize rays, CPU work or elapsed time. Source loading, transfer and
offline construction are additional costs, not free preprocessing. The current
shared-process runs do not establish cold-start speedup or amortization.

Use track-specific and per-map results first. Any pooled map/K mean is descriptive;
K values and random draws from the same map are not independent map samples.
Random seeds describe candidate-selection variability, not generalization.
No operational near-optimal tolerance or deployment acceptance threshold has
been supplied, so do not invent one as a pass/fail standard.

## Current Evidence And Expansion Rules

The completed reference matches the full optimum in 10/12 conditions for each
of library, union and portfolio. All six B conditions tie across deterministic
methods. A includes both gains and losses. Thus candidate reduction is working,
but consistent library-specific superiority is not established.

| Observation to investigate | Next isolated change, only when justified |
|---|---|
| Missing useful poses or installation-side allocations | Source geometry or layout diversity, not a simultaneous RL change |
| Performance depends on physical query budget | Explicit budget sweep with sources, maps and selection rules held fixed |
| Transfer degrades under a declared geometry or obstacle change | Separate geometry/occlusion stress test with all controls retained |
| Useful candidates exist but deployment feasibility changes | One specified installation/cost constraint with a common evaluator |
| Candidate quality is adequate but the selector cannot exploit it | Separate optimizer/refinement comparison with matched inputs and measured work |
| Quality matches but runtime benefit is unknown | Repeated fresh-process timing and offline-cost amortization study |

These are diagnostic hypotheses, not established causes of the observed losses.
Do not add complexity merely to obtain favorable library results. If strong
controls continue to match or outperform reuse, retain that conclusion and
reconsider the method. RL is optional, not the default explanation or remedy.

## Next Run Gate

Before the next comparison, create a separate versioned experiment specification:

1. Identify exactly one question and changed factor. Freeze sources, target
   geometry families, candidate/target grids, methods, seeds and budgets before
   inspecting the new target visibility or outcomes. Keep known maps as development
   cases; declare untouched families separately and report all of them.
2. For source-count or alternative-count comparisons, choose q independently of
   the transferred pool size. Freeze a geometry-only ranking if the pool exceeds
   q. If it is smaller, report the actual shortfall: do not pad it using target
   coverage. A geometry-filled hybrid needs its own declared method and control;
   unequal actual queries must not be labeled an exact matched-q comparison.
3. Count distinct layouts, not duplicate incumbent records. Keep source objective
   loss and branch/side allocation metadata; more records need not mean more
   diversity. Do not use a target optimum to select source alternatives.
4. Freeze query plans before physics and full-reference evaluation after arm
   decisions. Require installation checks, mask agreement, fresh saved-pose
   verification and exact-enumeration guards for the current small-domain design.
5. Run regression tests and dry-runs before actual experiments. Use new output
   roots and frozen snapshots; preserve failures. Inspect logs after completion,
   except process/timeout checks needed for execution safety.

**Readiness:** the existing single-source fixed-pool comparison is implemented
and complete. Multi-source retrieval, explicit pool-independent budgets in the
study runner, alternative-count filtering and new obstacle adapters are not
implemented by this decision. Exact next-run factor levels, new target manifest
and execution command are **TBD**. Do not launch an expansion under this document
or relabel the completed reference as a new experiment.

## Authorized Reference Repeat (2026-11-19)

After this planning decision, the user explicitly requested execution with no
interim analysis. The existing four-case protocol was rerun unchanged under
`results/library_query_controls_repeat_20261119/`, not expanded. Its `run_spec.json`
and `run.sh` identify the reproduction. All four runs and final report validation
completed; the original sources and results were preserved. Source-code hashes,
library hashes and target specifications matched the original reference before
launch. Logs and outcomes were inspected only after the batch exited successfully.

Preflight passed 131 tests (7 pipeline, 7 deployment-library, 117 library tests),
three legacy CLI dry-runs and four comparison dry-runs. All 408 raw comparison
rows, 12 condition-summary rows and target/mesh arrays matched the November 18
reference exactly. This is reproducibility evidence on the same development
maps, not an additional independent sample, runtime benchmark or new-factor
experiment. The next-run expansion specification remains TBD.
