# LiDAR 설치 최적화 실험 결과 정리

이 프로젝트는 도로 환경에서 LiDAR 설치 위치와 자세를 최적화하기 위해 `MILP`, `GREEDY`, `MAPPO` 세 방법을 비교한 실험 코드와 최종 결과를 담고 있습니다.

현재 기준으로 봐야 할 최종 결과 폴더는 하나입니다.

```text
results/journal_r30_20260513_183708/
```

과거 smoke run, scenario 0~3 임시 run, 로그, PNG figure는 정리되어 삭제되었습니다. 현재 남아 있는 결과는 최종 수치 분석용 JSON/NPY/CSV입니다.

## 현재 구조

```text
configs/
  experiment.yaml

data/
  scenarios/yeongjong_final.json
  cache/

src/
  environment.py
  outputs.py
  visualize.py
  methods/
    milp.py
    greedy.py
    mappo.py

results/
  journal_r30_20260513_183708/
    greedy/
    mappo/
    milp/
    tables/

main.py
test.py
view_result.py
run_journal_experiment.sh
resume_journal_experiment.sh
README.md
```

## 실험 설정

실험 설정 파일은 다음입니다.

```text
configs/experiment.yaml
```

주요 설정은 다음과 같습니다.

```yaml
lidar_profile: VELODYNE64_R30
scenario_ids: [0, 1, 2, 3, 6, 9]
k_values: [1, 2, 3, 4, 5, 6]
```

LiDAR 조건:

```yaml
VELODYNE64_R30:
  h_fov_deg: 360.0
  v_fov_deg: 26.9
  max_range_m: 30.0
```

후보 pose 조건:

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

Coverage target 조건:

```yaml
target:
  coverage_z_slices: [1.0, 1.5, 2.0]
  road_detection: 1
```

`road_detection: 1`은 target point 하나를 LiDAR 하나 이상이 보면 covered로 계산한다는 뜻입니다. 여러 LiDAR가 같은 target을 보더라도 coverage는 한 번만 카운트됩니다.

MILP solver 조건:

```yaml
solver:
  solver: gurobi
  threads: 1
  time_limit_sec: 1800
  mip_gap: 0.05
  nodefile_start_gb: 0.5
  nodefile_dir: data/gurobi_nodefiles
  soft_mem_limit_gb: 80
```

MAPPO 조건:

```yaml
mappo:
  episodes: 20000
  trials: 5
  seed_base: 0
```

## 실험 시나리오

시나리오는 `data/scenarios/yeongjong_final.json`에서 불러옵니다.

| scenario | topology | road width | lanes |
|---:|---|---:|---:|
| 0 | STRAIGHT | 7.0m | 2 |
| 1 | T_JUNCTION | 7.0m | 2 |
| 2 | CROSS | 7.0m | 2 |
| 3 | STRAIGHT | 21.0m | 6 |
| 6 | CROSS | 21.0m | 6 |
| 9 | T_JUNCTION | 28.0m | 8 |

각 scenario에 대해 K = 1, 2, 3, 4, 5, 6을 실험했습니다.

## 공통 환경

모든 알고리즘은 `src/environment.py`를 공통 환경으로 사용합니다.

이 파일이 담당하는 기능:

```text
config 로드
scenario 로드
도로/인도/건물/중앙선 geometry 생성
LiDAR 설치 가능 영역 판정
도로 target point 생성
Open3D raycast visibility cache 생성
최종 coverage 평가
```

중요한 점은 최종 coverage가 모두 같은 함수로 계산된다는 것입니다.

```text
src.environment.evaluate_coverage()
```

따라서 MILP, GREEDY, MAPPO 결과는 같은 map, 같은 LiDAR profile, 같은 target, 같은 visibility 기준에서 비교됩니다.

## 알고리즘 구성

### MILP

파일:

```text
src/methods/milp.py
```

역할:

```text
후보 pose 생성
후보별 coverage mask 계산
Gurobi MILP 모델 생성
at-most-K maximum coverage 문제 해결
결과 JSON/NPY 저장
```

MILP formulation:

```text
x_i = 후보 LiDAR i 선택 여부
y_j = target j covered 여부

maximize sum_j y_j
subject to y_j <= sum_i a_ij x_i
           sum_i x_i <= K
           x_i, y_j binary
```

현재 formulation은 exact-K가 아니라 `at-most-K`입니다. 즉 K개 이하의 LiDAR로 coverage를 최대화합니다.

### GREEDY

파일:

```text
src/methods/greedy.py
```

역할:

```text
MILP와 같은 후보 set 사용
현재 미커버 target을 가장 많이 새로 덮는 후보를 순차 선택
결과 JSON/NPY 저장
```

GREEDY는 학습이 없고 deterministic입니다.

### MAPPO

파일:

```text
src/methods/mappo.py
```

역할:

```text
legacy MAPPO native port
K개의 LiDAR agent 학습
11-action policy 사용
PPO 학습
trial별 JSON/NPY/train_log 저장
최종 coverage를 공통 evaluator로 재평가
```

MAPPO action space:

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

MAPPO는 각 scenario/K마다 5 trials를 수행했습니다.

## 최종 결과 폴더

최종 결과는 다음 폴더에 있습니다.

```text
results/journal_r30_20260513_183708/
```

구성:

```text
greedy/
  GREEDY 결과 JSON/NPY

mappo/
  MAPPO trial별 JSON/NPY/train_log_npz

milp/
  MILP 결과 JSON/NPY

tables/
  compare_methods.csv
```

파일 개수:

```text
greedy: 36 JSON + 36 NPY
mappo: 180 JSON + 180 NPY + 180 train_log NPZ
milp: 36 JSON + 36 NPY
tables: compare_methods.csv
```

## 최종 비교 CSV

가장 중요한 결과 파일은 다음입니다.

```text
results/journal_r30_20260513_183708/tables/compare_methods.csv
```

CSV row 구성:

```text
MILP_BUDGET: 36 rows
GREEDY: 36 rows
MAPPO: 180 rows
MAPPO_MEAN: 36 rows
```

즉 MAPPO는 trial별 결과 180개가 있고, scenario/K별 평균인 `MAPPO_MEAN` 36개가 추가로 들어 있습니다.

주요 column:

```text
scenario_id
method
K
coverage_pct
trial
seed
mappo_tag
std_coverage_pct
raw_coverage_pct
solver_status
mip_gap
result_file
```

논문 표나 그래프를 만들 때는 보통 다음 세 method를 비교하면 됩니다.

```text
MILP_BUDGET
GREEDY
MAPPO_MEAN
```

## MILP solver 상태

MILP는 총 36개 문제를 모두 풀었지만, 전부 optimal proof가 있는 것은 아닙니다.

```text
solver_status 2  = optimal: 19개
solver_status 9  = time limit: 10개
solver_status 17 = memory limit: 7개
```

따라서 MILP 결과를 논문에 쓸 때는 `solver_status`와 `mip_gap`을 함께 보고해야 합니다.

주의:

- `solver_status = 2`: Gurobi가 optimal로 종료한 결과
- `solver_status = 9`: time limit에서 저장된 feasible result
- `solver_status = 17`: memory limit에서 저장된 feasible result

일부 time/memory limit 결과는 `mip_gap = inf`로 저장되어 있습니다. 이 경우 optimality proof가 없으므로 “best feasible under solver limit”로 해석해야 합니다.

## 결과 파일 형식

각 알고리즘 폴더의 JSON 파일은 metadata와 coverage 결과를 담고 있습니다.

예:

```text
method
scenario_id
topology
lidar_profile
budget_k
coverage_pct
covered_count
num_targets
positions_npy
config_hash
cache_key
```

MILP JSON에는 추가로 다음이 있습니다.

```text
solver_status
solver_backend
mip_gap
raw_coverage_pct
frontier_coverage_pct
```

MAPPO JSON에는 추가로 다음이 있습니다.

```text
episodes
seed
trial
mappo_tag
train_log_npz
```

NPY 파일은 실제 선택된 LiDAR pose입니다.

```text
[x, y, z, yaw, pitch]
```

MAPPO의 `*_train_log.npz`는 episode별 학습 기록입니다.

## 결과 확인 방법

### CSV 확인

```bash
python - <<'PY'
import pandas as pd
p = 'results/journal_r30_20260513_183708/tables/compare_methods.csv'
df = pd.read_csv(p)
print(df.groupby('method').size())
print(df[df['method'].isin(['MILP_BUDGET', 'GREEDY', 'MAPPO_MEAN'])].head())
PY
```

### 3D viewer로 특정 결과 보기

```bash
.venv/bin/python view_result.py \
  --config configs/experiment.yaml \
  --results-root results/journal_r30_20260513_183708 \
  --method mappo \
  --scenario-id 0 \
  --k 1 \
  --mappo-tag journal_r30_20260513_183708 \
  --viewer matplotlib
```

특정 JSON을 바로 열 수도 있습니다.

```bash
.venv/bin/python view_result.py \
  --config configs/experiment.yaml \
  --result-json results/journal_r30_20260513_183708/milp/파일명.json \
  --viewer matplotlib
```

## figure 재생성

과거 PNG figure는 정리 과정에서 삭제했습니다. 다시 만들려면 아래 명령을 실행하면 됩니다.

```bash
.venv/bin/python main.py visualize \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --methods milp,greedy,mappo \
  --mappo-tag journal_r30_20260513_183708 \
  --results-root results/journal_r30_20260513_183708 \
  --output-dir results/journal_r30_20260513_183708/figures \
  --output-3d-dir results/journal_r30_20260513_183708/3d \
  --save-3d
```

생성되는 폴더:

```text
results/journal_r30_20260513_183708/figures/
results/journal_r30_20260513_183708/3d/
```

## 재실행 방법

전체 실험을 새로 실행하려면:

```bash
./run_journal_experiment.sh
```

중간에 끊긴 실험을 이어서 실행하려면:

```bash
./resume_journal_experiment.sh
```

단, 현재 최종 결과는 이미 생성되어 있으므로 보통은 재실행할 필요가 없습니다.

## 현재 결론

현재 실험은 다음까지 완료된 상태입니다.

```text
GREEDY 완료
MAPPO 완료
MILP 완료
compare_methods.csv 생성 완료
PNG figure는 삭제됨. 필요 시 재생성 가능
```

분석의 기준 파일은 다음 하나입니다.

```text
results/journal_r30_20260513_183708/tables/compare_methods.csv
```
