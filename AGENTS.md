# AGENTS.md

## Project Purpose

This repository contains research code for 3D LiDAR placement optimization in parametric road environments. It compares MILP, Greedy, and MAPPO-style learning methods under a shared scenario, candidate generation, and coverage-evaluation pipeline.

Treat this as an AI / robotics research portfolio repository. Future Codex work should improve reproducibility, documentation clarity, and evidence quality without inventing results.

## Coding Style Rules

- Preserve current experiment behavior unless the user explicitly requests a refactor.
- Keep method implementations separated under `src/methods/`.
- Keep shared geometry, LiDAR profile, visibility cache, and coverage logic in `src/environment.py`.
- Prefer typed, explicit data structures and small helper functions over implicit global state.
- Keep CLI behavior in `main.py` stable when changing internals.
- Do not silently change scenario IDs, K budgets, LiDAR profiles, or solver defaults.

## Documentation Rules

- Document the exact command that generates any claimed result.
- Distinguish dry-run/planning output from solved optimization results.
- Mark missing results, figures, tables, or trained policies as `TBD` or `To be added`.
- Explain solver requirements clearly, especially Gurobi license requirements and fallback behavior.
- Keep descriptions technical and concise for AI / robotics engineering reviewers.

## README Writing Rules

- Start with the research question and method comparison.
- Include a repository map, quick-start commands, expected outputs, and validation status.
- Do not claim quantitative superiority unless committed result files or reproducible logs are present.
- When results are not committed, say so explicitly and provide the command path to regenerate them.
- Prefer command blocks over vague instructions.

## Testing / Validation Rules

- Run syntax checks when Python is available:
  - `python -m py_compile main.py test.py view_result.py src/environment.py src/outputs.py src/visualize.py src/methods/milp.py src/methods/greedy.py src/methods/mappo.py tests/test_pipeline.py`
- Run lightweight pipeline tests when dependencies are installed:
  - `python tests/test_pipeline.py`
- Run dry-run checks before expensive experiments:
  - `python main.py milp --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run`
  - `python main.py greedy --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run`
  - `python main.py mappo --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --episodes 10 --trials 1 --tag dryrun --dry-run`
- If dependencies such as Open3D, PyTorch, Gurobi, or a Python runtime are unavailable, state exactly what could not be validated.

## What Codex Must Not Fabricate

- Do not invent coverage scores, training curves, runtime numbers, solver outcomes, or method rankings.
- Do not claim MAPPO converges unless committed logs or generated artifacts prove it.
- Do not claim a scenario came from real OSM geometry if the code currently generates a parametric road environment from scenario metadata.
- Do not add academic publication, thesis, or benchmark claims without source evidence.

## Missing Results / Dataset Handling

- Keep generated outputs under ignored paths such as `results/`, `logs/`, `data/cache/`, and `data/gurobi_nodefiles/`.
- If a result is missing, write `TBD` and add the exact command expected to produce it.
- If a dataset cannot be included, document the schema and small sample metadata rather than fabricating data.
- Do not commit large `.npy`, `.npz`, cache, or solver node files unless the user explicitly asks for curated artifacts.

## Portfolio-Quality Explanation Rules

- Explain the system as: scenario metadata -> parametric road geometry -> LiDAR candidate generation -> visibility/cache -> method-specific selection -> shared coverage evaluation -> JSON/CSV/figure outputs.
- Highlight engineering constraints: solver licensing, candidate explosion, visibility cache cost, GPU/CPU requirements for MAPPO, and reproducibility boundaries.
- Keep claims evidence-backed and separate implemented code from planned result assets.

