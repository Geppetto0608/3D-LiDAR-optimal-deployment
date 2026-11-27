# Geometry Selection Mechanism Study

Protocol date: 2026-11-09, Asia/Seoul. Gate 1 of `TITS_RESEARCH_PLAN.md`.
This is a mechanism check on known development cases, not a novelty proof,
network-scale deployment demonstration or submission-readiness assessment.

## Question And Frozen Scope

Does measured visibility feedback improve the existing geometry selector beyond
what optical pair interactions alone can achieve at the same pose-query budget?

Keep `configs/geometry_placement.json` and `configs/connected_corridor_pilot.json`
unchanged: six known bent-corridor geometries, base/combined profiles, fixed K=2,
the original geometry/candidate/target grids, cost/coverage/overlap objective J
and installation/interface/failure constraints. These are synthetic illustrative
requirements, not operational RSU standards. Detection and MAPPO remain excluded.
Mirrored shapes and constraint profiles are not independent map families.

Budgets are 1, 2 and 4 actual candidate-pose queries per eligible installation
site. Use three timing repetitions, independent fresh processes with cyclic arm
ordering. Four limited methods plus a full reference give 13 arms per case and
12 x 13 x 3 = 468 workers. Repetitions do not increase the quality sample size.
Failures stop the batch; do not replace cases or thresholds after seeing results.

## Controls

Let g_c be the unoccluded range/FOV mask, a_c the physically measured mask,
and J(c,d; m_c,m_d) the existing common pair objective with the existing
pair feasibility checks. Surrogate feasibility is NOT physical feasibility.

1. `static`: existing single-candidate geometry ranking, unchanged.
2. `static_pair`: score every eligible candidate by its best predicted feasible
   partner at a different site, using (g_c,g_d) for both masks. Rank candidates
   at each site once; take the first q. Ties use rounded pair J (8 decimals),
   then the existing geometry prior, then candidate ID. A candidate with no
   predicted feasible partner falls back behind candidates with such partners,
   ordered by geometry prior and ID. The best partner need not be selected into
   the actual query pool. This is a stronger ranking control, not an exact
   optimum of the joint query-pool selection problem.
3. `no_feedback`: reuse the adaptive selector's exact seeding, synchronous
   rounds, partner eligibility, fallback and tie rules, but supply g_d in place
   of a_d for the already planned partner IDs. Freeze the entire sequence before
   the first physical query. No measured masks or physical callback are passed
   to the planning function. Diverging later contexts are an effect of removing
   feedback; we do not force this control to follow an oracle adaptive trace.
4. `adaptive_batch`: existing selector with predicted unqueried mask g_c and
   actually measured partner mask a_d. The shared schedule is extracted without
   changing its decision rules. The scalar implementation remains available.
5. `full`: query every eligible candidate; solve the same finite K=2 problem.
   Its masks and solutions are not accessible to the limited methods.

Every method uses actual queried masks for the final exact pair selection and
directly verifies the selected deployment. Predicted masks cannot certify the
final objective or constraints. Optical computations, virtual planning masks,
directed candidate-partner tests and actual physical rays/queries are distinct.
Directed pair tests count considered pairs before screening, not the number of
expensive full evaluator calls. All planning overhead is included in latency.

## Primary Comparisons

- `adaptive_batch - no_feedback`: effect of replacing predicted partner masks
  with measurements under the same schedule, domain and query budget.
- `static_pair - static`: gain from a stronger optical pair-aware ranking.
- `adaptive_batch - static_pair`: performance against that stronger static control;
  this comparison alone does not isolate feedback causally.

Report feasibility first, then paired J differences and full-reference loss for
comparable feasible cases. Infeasible J is null, not zero; failure to find a
feasible queried pair is not a global infeasibility proof. Include every case,
negative result and denominator. J units are not coverage percentage points.

Report actual pose queries, rays and quality-query curves, plus cold-ready and
whole-worker time using per-case repetition medians before equal-weight averaging.
Cold-ready includes import/input/geometry/selection/optimization but excludes
final direct verification and export; whole-worker time includes those phases.
These remain shared-server measurements without CPU affinity or OS cache eviction.
No equal-ray, equal-FLOP, controlled speedup or statistical significance claim.

## Validation Before Interpretation

- Exhaustively compare static pair scores/partners with the full common evaluator
  on small randomized masks, costs, constraints, infeasibility and ties.
- Perturb physical masks and source scores: both control plans must stay unchanged.
- Assert the complete control plan exists before the first physical callback.
- If predicted and actual masks agree, no-feedback must reproduce adaptive events,
  contexts and decisions. Budget 1 must agree even with occlusion.
- Verify scalar/batch parity, nested budget pools, eligibility, empty forbidden
  domains, strict budgets and separate virtual/physical accounting.
- Real fresh-worker smoke including a completely blocked installation domain;
  independent pair enumeration, direct raycast and rehashed plan-tamper rejection.
- Existing hierarchy/prior/response/pipeline regression, syntax and legacy dry runs.
- Full-run frozen input/source/output hashes, independent finite-pair optima,
  policy replay, fresh physical masks, sampled independent triangle checks,
  deterministic repeats and table reconstruction.
- Preserve historical artifacts and compare unchanged adaptive/static/full
  decisions/quality with the previous equal-query study.

Policy replay shares production selection code. Small exhaustive pair checks and
physical samples provide additional checks, not an independent proof of every ray.

## Reproduction

The plan command omits `--execute` and creates no results. Actual run, new root:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py \
  --methods adaptive_batch static static_pair no_feedback full \
  --query-budgets 1 2 4 --repeats 3 \
  --output results/geometry_mechanism_20261109 --execute
.venv/bin/python results/geometry_mechanism_20261109/source/geometry_mechanism_report.py \
  results/geometry_mechanism_20261109
```

The main runner performs its frozen audit after all workers finish. The separate
report only reads audited artifacts. Results are ignored and absent from a clone.
`mechanism_report/paired.csv` contains the three contrasts; `selection_work.csv`
separates planning time and virtual work; `summary.json` contains all methods and
actual denominators. `provenance.json` identifies the inputs and report outputs.

## Outcome

Completed on 2026-11-09: 468/468 workers, all feasible; 75 regression tests,
syntax, legacy CLI plans, independent finite-pair/mask/timing audits passed.
The experiment snapshot of this protocol retains its pre-run TBD status;
this section is the post-run interpretation and does not modify that snapshot.

| Queries/site | Adaptive mean J loss | Static mean J loss | Static pair mean J loss | No-feedback mean J loss |
|---:|---:|---:|---:|---:|
| 1 | 7.075 | 7.075 | 4.925 | 7.075 |
| 2 | 1.183333 | 6.675 | 0.525 | 1.183333 |
| 4 | 0 | 4.341667 | 0 | 0 |

Adaptive and no-feedback have identical final sensor IDs and J in all 36
case/budget combinations. Their query pools differ in 8/12 q2 and 10/12 q4
cases: feedback is active, but provides no final deployment benefit here.
At q2 static-pair beats adaptive in four conditions, tying eight. At q1 it also
loses to static-single in two conditions; do not claim uniform dominance.
At q4 adaptive/static-pair/no-feedback each match full in 12/12 conditions,
but their mean per-case median cold-ready times (3.127/3.147/3.133 s) exceed
full (2.879 s). Whole-worker means show the same direction. Query reduction is
not an end-to-end speed advantage. These are shared-server descriptive timings.

**Decision:** do not promote measured-feedback benefit as established novelty.
Keep pair-aware source-free controls in subsequent studies. Next investigate
whether cheap, valid partial-visibility bounds are useful on tiny problems,
before expanding to network scale. A new name for pair ranking is not a new
contribution; no detector or MAPPO was added to rescue the claim.

[Full results](results/geometry_mechanism_20261109/REPORT.md),
[paired contrasts](results/geometry_mechanism_20261109/mechanism_report/paired.csv),
[saved-mask diagnostics](results/geometry_mechanism_checks_20261109/feedback_diagnostics.json),
[independent audit](results/geometry_mechanism_20261109/independent_validation.json).
The unchanged adaptive/static/full decisions, masks and deployments match 84
historical case/arms across 252 new timing workers; 6,207 historical file hashes
were preserved. Independent arithmetic checked 156 case/arms and 108 contrasts.
