# 3D LiDAR Optimal Deployment

도로 환경에서 3D LiDAR의 설치 위치와 자세를 최적화하기 위한 연구 코드입니다.

이 저장소는 `MILP`, `GREEDY`, `MAPPO` 세 방법을 같은 환경 조건에서 비교하기 위한 실험 파이프라인을 포함합니다. 현재 저장소에는 코드, 설정 파일, 시나리오 메타데이터가 포함되어 있으며, 실험 결과물(`results/`)은 포함하지 않습니다.


## Repository Status

현재 상태는 실험을 다시 실행할 수 있는 코드베이스입니다.

```text
configs/experiment.yaml              실험 설정
data/scenarios/yeongjong_final.json  도로 시나리오 메타데이터
main.py                              통합 CLI
test.py                              짧은 실행/점검용 runner
view_result.py                       결과 3D viewer
src/environment.py                   공통 환경, 지도, raycast, coverage 계산
src/outputs.py                       결과 JSON/CSV 처리
src/visualize.py                     figure 생성
src/methods/milp.py                  MILP 최적화
src/methods/greedy.py                GREEDY baseline
src/methods/mappo.py                 MAPPO 학습
tests/test_pipeline.py               파이프라인 테스트
```

`results/`, `data/cache/`, `logs/`, `.venv/`는 git에 포함하지 않습니다.

## Quick Start

```bash
git clone https://github.com/Geppetto0608/3D-LiDAR-optimal-deployment.git
cd 3D-LiDAR-optimal-deployment
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lightweight checks:

```bash
python -m py_compile main.py test.py view_result.py src/environment.py src/outputs.py src/visualize.py src/methods/milp.py src/methods/greedy.py src/methods/mappo.py tests/test_pipeline.py
python tests/test_pipeline.py
```

Dry-run examples:

```bash
python main.py milp --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run
python main.py greedy --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run
python main.py mappo --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --episodes 10 --trials 1 --tag dryrun --dry-run
```

Full runs can be expensive. Gurobi requires a valid license; if unavailable, the code can fall back to PuLP/CBC for smaller cases.

## Experiment Goal

목표는 도로 target coverage를 최대화하는 LiDAR 배치 방법을 비교하는 것입니다.

비교 알고리즘:

- `MILP`: at-most-K maximum coverage 최적화
- `GREEDY`: marginal coverage gain 기반 순차 선택
- `MAPPO`: multi-agent PPO 기반 LiDAR pose 탐색

모든 알고리즘은 같은 LiDAR profile, 같은 도로 target, 같은 visibility/coverage 계산을 사용합니다.

## Environment

공통 환경은 [src/environment.py](src/environment.py)에 있습니다.

현재 map은 OSM 원본 도로 형상을 직접 사용하는 방식이 아니라, 시나리오의 `topology`, `road_width`, `num_lanes`를 기반으로 생성되는 100 m x 100 m parametric road environment입니다.

환경 구성:

- 도로 영역
- 인도 및 설치 가능 영역
- 중앙선/중앙분리대
- 건물 또는 설치 불가 영역
- 도로 위 coverage target points
- Open3D 기반 raycast visibility cache

Coverage target은 도로 위 grid point에 높이 slice를 적용하여 생성합니다.

```yaml
target:
  coverage_z_slices: [1.0, 1.5, 2.0]
  road_detection: 1
```

`road_detection: 1`은 target 하나를 LiDAR 하나 이상이 보면 covered로 계산한다는 뜻입니다. 여러 LiDAR가 같은 target을 보더라도 coverage는 한 번만 카운트됩니다.

## LiDAR Model

기본 profile은 `VELODYNE64_R30`입니다.

```yaml
VELODYNE64_R30:
  h_fov_deg: 360.0
  v_fov_deg: 26.9
  max_range_m: 30.0
```

LiDAR pose는 다음 5개 값으로 표현합니다.

```text
[x, y, z, yaw, pitch]
```

Roll은 사용하지 않습니다.

설치 후보 설정:

```yaml
candidate:
  xy_step_m: 2.0
  z_min: 3.0
  z_max: 6.0
  z_step: 1.0
  yaw_step_deg: 5.0
  yaw_axis_min_deg: 35.0
  yaw_axis_max_deg: 145.0
  pitch_min_deg: -25.0
  pitch_max_deg: -5.0
  pitch_step_deg: 2.5
  max_orientations_per_position: 0
```

## Scenarios

시나리오는 [data/scenarios/yeongjong_final.json](data/scenarios/yeongjong_final.json)에서 불러옵니다.

현재 기본 실험 설정:

```yaml
scenario_ids: [0, 1, 2, 3, 6, 9]
k_values: [1, 2, 3, 4, 5, 6]
```

대표 시나리오:

| scenario | topology | road width | lanes |
|---:|---|---:|---:|
| 0 | STRAIGHT | 7.0 m | 2 |
| 1 | T_JUNCTION | 7.0 m | 2 |
| 2 | CROSS | 7.0 m | 2 |
| 3 | STRAIGHT | 21.0 m | 6 |
| 6 | CROSS | 21.0 m | 6 |
| 9 | T_JUNCTION | 28.0 m | 8 |

## Algorithms

### MILP

파일: [src/methods/milp.py](src/methods/milp.py)

MILP는 가능한 LiDAR candidate pose를 먼저 생성한 뒤, 최대 K개를 선택해 covered target 수를 최대화합니다.

변수:

```text
x_i = candidate i 선택 여부
y_j = target j covered 여부
a_ij = candidate i가 target j를 cover하면 1
```

Formulation:

```text
maximize   sum_j y_j
subject to y_j <= sum_i a_ij x_i
           sum_i x_i <= K
           x_i, y_j binary
```

현재 formulation은 `exact-K`가 아니라 `at-most-K`입니다.

### GREEDY

파일: [src/methods/greedy.py](src/methods/greedy.py)

GREEDY는 MILP와 같은 candidate set을 사용합니다. 매 step마다 아직 covered 되지 않은 target을 가장 많이 새로 덮는 candidate를 선택합니다.

```text
choose candidate with maximum marginal coverage gain
update covered target set
repeat until K selections or no positive gain
```

학습이 없는 deterministic baseline입니다.

### MAPPO

파일: [src/methods/mappo.py](src/methods/mappo.py)

MAPPO는 K개의 LiDAR를 K개의 agent로 보고, 각 agent가 action을 통해 pose를 이동하며 coverage를 높이도록 학습합니다.

Action space:

```text
0  stay
1  x+
2  x-
3  y+
4  y-
5  yaw+
6  yaw-
7  pitch+
8  pitch-
9  z+
10 z-
```

학습 후 최종 pose는 공통 coverage evaluator로 다시 평가하여 JSON/NPY로 저장합니다.

## CLI

통합 실행 파일은 [main.py](main.py)입니다.

지원 subcommand:

```text
milp
greedy
mappo
compare
visualize
full
```

Dry-run 예시:

```bash
.venv/bin/python main.py milp \
  --config configs/experiment.yaml \
  --scenario-ids 0 \
  --k-values 1 \
  --dry-run

.venv/bin/python main.py greedy \
  --config configs/experiment.yaml \
  --scenario-ids 0 \
  --k-values 1 \
  --dry-run

.venv/bin/python main.py mappo \
  --config configs/experiment.yaml \
  --scenario-ids 0 \
  --k-values 1 \
  --episodes 10 \
  --trials 1 \
  --tag dryrun \
  --dry-run
```

실제 실행은 계산량이 큽니다. 특히 MILP는 Gurobi license와 메모리/시간 제한의 영향을 받습니다.

## Outputs

실험 결과는 기본적으로 `results/active/` 아래에 저장됩니다.

예상 구조:

```text
results/active/
  milp/
    *.json
    *.npy
  greedy/
    *.json
    *.npy
  mappo/
    *.json
    *.npy
    *_train_log.npz
  tables/
    compare_methods.csv
  figures/
  3d/
```

결과 파일은 git에 포함하지 않습니다.

## Validation

기본 문법 검사는 다음 명령으로 수행합니다.

```bash
.venv/bin/python -m py_compile \
  main.py test.py view_result.py \
  src/environment.py src/outputs.py src/visualize.py \
  src/methods/milp.py src/methods/greedy.py src/methods/mappo.py \
  tests/test_pipeline.py
```

Current committed validation evidence:

- Pipeline smoke tests are included in [tests/test_pipeline.py](tests/test_pipeline.py).
- Generated result tables, figures, solver logs, and MAPPO training curves are not committed.
- Quantitative comparison between MILP, GREEDY, and MAPPO: TBD.
- Runtime and memory benchmark: TBD.

## Portfolio TODO

Highest-impact additions before a portfolio review:

1. Add a small curated `docs/` figure showing one scenario, LiDAR placements, and covered/uncovered targets.
2. Add one reproducible small-case result table generated from a documented command.
3. Add an architecture diagram: scenario metadata -> candidate generation -> visibility cache -> solver/agent -> shared evaluator -> outputs.
4. Document hardware/software environment used for any full experiment results.
5. Add a note explaining which results require Gurobi and which can run with PuLP/CBC.

## Citation and Use

이 저장소의 코드, 구조, 실험 설정, 결과물은 저자의 명시적 허가 없이 학술 논문, 학위논문, 기술보고서, 상업적 산출물에 사용할 수 없습니다.

사용 허가가 필요한 경우 repository owner에게 별도로 문의해야 합니다.
