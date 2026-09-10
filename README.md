# 3D LiDAR Optimal Deployment

도로 환경에서 3D LiDAR의 설치 위치, 높이, yaw, pitch를 최적화하고 `MILP`, `GREEDY`, `MAPPO`를 동일한 visibility/coverage 기준으로 비교하기 위한 연구 코드입니다.

이 문서는 다음 내용을 한 번에 확인할 수 있도록 작성되었습니다.

- 연구 목적과 비교 원칙
- 프로젝트 전체 파일 구조
- 시나리오와 100 m x 100 m 도로 환경 생성 방법
- LiDAR pose, FOV, 설치 가능 영역, target 정의
- Open3D raycasting cache와 coverage 계산식
- MILP, GREEDY, MAPPO의 실제 코드 구성과 수식
- 실행 순서, 결과 JSON/NPY/NPZ/CSV 형식
- 현재 완료 결과와 시나리오별 수치
- 코드 및 결과 무결성 검증 결과
- 논문에 사용하기 전에 반드시 해결해야 하는 한계

> 검토 기준일: 2026-08-07
>
> 현재 완료 결과: `results/journal_r30_top10_20260707/`
>
> 중요: 현재 MAPPO 결과 파일의 coverage 계산은 정확하지만 PPO 학습 구현 오류가 확인되어 잠정 결과로만 취급해야 합니다. 자세한 내용은 [논문 사용 전 필수 수정사항](#논문-사용-전-필수-수정사항)에 정리되어 있습니다.

## 1. 연구 목적

목표는 도로 위 검지 대상 지점을 최대한 많이 관측하도록 제한된 수의 LiDAR를 배치하는 것입니다.

LiDAR 한 대의 pose는 다음 5개 변수로 표현합니다.

```text
p_i = [x_i, y_i, z_i, yaw_i, pitch_i]
```

현재 모델에는 roll이 없습니다.

비교 방법은 다음과 같습니다.

| 방법 | 역할 | 학습 여부 | 주요 출력 |
|---|---|---:|---|
| MILP | 이산 후보 중 최대 K개를 선택하는 maximum coverage 최적화 | 없음 | feasible solution, solver status, MIP gap |
| GREEDY | 매 단계 marginal coverage gain이 가장 큰 후보 선택 | 없음 | deterministic baseline |
| MAPPO | K개 LiDAR agent가 pose를 이동하며 PPO로 학습 | 있음 | trial별 최종 pose와 학습 로그 |

세 방법의 최종 coverage는 모두 `src.environment.evaluate_coverage()`와 동일한 target, visibility cache, FOV 판정을 사용합니다.

다만 현재 MILP/GREEDY는 위치별 상위 10개 orientation만 사용하고 MAPPO는 전체 yaw/pitch 범위를 행동으로 탐색합니다. 따라서 현재 MILP는 MAPPO 전체 행동 공간에 대한 엄밀한 upper bound가 아닙니다.

## 2. 현재 프로젝트 구조

```text
hj_ws/
├── configs/
│   └── experiment.yaml
├── data/
│   ├── scenarios/
│   │   └── yeongjong_final.json
│   ├── cache/                   # raycasting visibility cache, git 제외
│   ├── gurobi_nodefiles/        # Gurobi 임시 node file, git 제외
│   └── osm/                     # 현재 계산에는 사용하지 않음
├── src/
│   ├── environment.py           # 설정, 지도, LiDAR, raycasting, coverage
│   ├── outputs.py               # JSON/NPY 저장, 로드, 비교 CSV
│   ├── visualize.py             # top-view, heatmap, 3D snapshot, 학습 그래프
│   └── methods/
│       ├── milp.py              # 후보 생성, MILP, budget frontier
│       ├── greedy.py            # greedy baseline
│       └── mappo.py             # native MAPPO port와 학습 환경
├── tests/
│   └── test_pipeline.py         # 경량 파이프라인 무결성 테스트
├── main.py                      # 통합 CLI
├── test.py                      # 안전 실행 runner, --execute 없으면 계획만 출력
├── view_result.py               # Matplotlib/Open3D 대화형 결과 viewer
├── README.md
└── LICENSE
```

로컬에는 다음 완료 결과가 존재합니다.

```text
results/journal_r30_top10_20260707/
├── greedy/
├── mappo/
├── milp/
├── tables/
├── figures/
│   └── comparison/
└── 3d/
```

`results/journal_r30_20260707/`은 GREEDY만 포함된 초기 중간 실행 폴더입니다. 최종 분석 기준으로 사용하지 않습니다.

## 3. 전체 파이프라인

```text
configs/experiment.yaml
        +
data/scenarios/yeongjong_final.json
        │
        ▼
src/environment.py
  ├─ parametric road geometry 생성
  ├─ coverage target 생성
  ├─ 설치 가능 위치 판정
  ├─ Open3D raycasting visibility cache 생성/로드
  └─ 공통 coverage evaluator 제공
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
 MILP 후보 최적화   GREEDY 순차 선택   MAPPO pose 학습
        │               │                │
        └───────────────┴────────────────┘
                        │
                        ▼
             공통 evaluator로 최종 계산
                        │
                        ▼
       JSON metadata + NPY pose + MAPPO NPZ log
                        │
                        ▼
        compare_methods.csv + figure + 3D viewer
```

실행 단위는 기본적으로 다음 조합입니다.

```text
scenario × K × method
```

MAPPO는 여기에 trial 차원이 추가됩니다.

```text
scenario × K × trial
```

현재 기본 설정은 시나리오 6개, K 6개, MAPPO trial 5개이므로 다음 작업 수를 가집니다.

```text
MILP:    6 × 6     = 36
GREEDY:  6 × 6     = 36
MAPPO:   6 × 6 × 5 = 180
```

## 4. 실험 설정

실제 설정 파일은 `configs/experiment.yaml`입니다.

### 4.1 기본 실험 범위

```yaml
lidar_profile: VELODYNE64_R30
scenario_ids: [0, 1, 2, 3, 6, 9]
k_values: [1, 2, 3, 4, 5, 6]
experiment_tag: r30_refactor
```

### 4.2 설치 후보 설정

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
  max_orientations_per_position: 10
```

허용 높이는 다음과 같습니다.

```text
z ∈ {3, 4, 5, 6} m
```

pitch 후보는 다음 9개입니다.

```text
pitch ∈ {-25, -22.5, -20, -17.5, -15, -12.5, -10, -7.5, -5} deg
```

yaw는 설치 위치에서 가장 가까운 도로 방향의 normal을 기준으로 제한합니다.

```text
relative yaw range = [35 - 90, 145 - 90]
                   = [-55, +55] deg
```

5도 간격이므로 일반적인 위치에서 상대 yaw 후보는 23개입니다.

```text
raw orientations per xyz position = 23 yaw × 9 pitch = 207
```

MILP/GREEDY는 207개를 모두 사용하지 않고 각 xyz 위치에서 단일 센서 coverage가 높은 상위 10개만 남깁니다. 이후 coverage mask가 완전히 같은 후보는 전역 deduplication합니다.

### 4.3 Coverage target 설정

```yaml
target:
  coverage_z_slices: [1.0, 1.5, 2.0]
  road_detection: 1
```

각 도로 XY cell에 높이 1.0 m, 1.5 m, 2.0 m target을 생성합니다.

`road_detection: 1`은 target을 하나 이상의 LiDAR가 보면 covered로 판정한다는 의미입니다. 여러 LiDAR가 같은 target을 보더라도 최종 coverage에는 한 번만 포함됩니다.

### 4.4 MILP solver 설정

```yaml
solver:
  solver: gurobi
  threads: 1
  time_limit_sec: 300
  mip_gap: 0.05
  nodefile_start_gb: 0.5
  nodefile_dir: data/gurobi_nodefiles
  soft_mem_limit_gb: 80
```

현재 문제당 제한시간은 300초이고 목표 relative MIP gap은 5%입니다. 이 설정은 빠른 feasible solution 확보를 위한 것이며 모든 문제의 exact optimum을 보장하지 않습니다.

### 4.5 MAPPO 설정

설정 파일에 명시된 PPO 값은 다음과 같습니다.

```yaml
mappo:
  episodes: 20000
  trials: 5
  seed_base: 0
  ppo_lr: 0.00001
  ppo_gamma: 0.999
  ppo_eps_clip: 0.15
  ppo_k_epochs: 5
  ppo_update_timestep: 512
  ppo_entropy_coef: 0.003
  ppo_gae_lambda: 0.95
  ppo_value_clip: 0.2
  ppo_minibatch_size: 128
```

코드의 나머지 실제 기본값은 다음과 같습니다.

| 항목 | 값 | 의미 |
|---|---:|---|
| `local_map_size` | 7 | agent local visibility map 크기 |
| `score_base_voxel` | 0.01 | target 하나의 reward weight |
| `warm_start_enabled` | true | visibility 상위 spawn 사용 |
| `spawn_top_k` | 200 | warm-start spawn 후보 수 |
| `w_abs_coverage` | 0.1 | 절대 coverage score 보상 계수 |
| `w_overlap_pen` | 0.0 | 중복 관측 penalty 계수 |
| `w_far_dist` | 1.0 | agent 간 거리 보상 계수 |
| `w_uncovered_move` | 1.5 | 미커버 영역 접근 보상 계수 |
| `dist_reward_scale` | 20.0 | 거리 정규화 값 |
| `w_marginal_gain` | 1.0 | agent 고유 기여 보상 계수 |
| `w_outward_pen` | 2.0 | 지도 경계에서 바깥쪽을 보는 penalty |
| `edge_margin_m` | 5.0 | 경계 penalty 적용 거리 |
| `ppo_entropy_min` | 0.0 | entropy coefficient 하한 |
| `ppo_entropy_decay` | 0.997 | update별 entropy decay |
| `ppo_random_action_prob` | 0.005 | 추가 random action 확률 |
| `eval_interval` | 100 | deterministic 평가 간격 |
| `eval_steps` | 5 | 평가 시 deterministic 이동 수 |
| `plateau_patience` | 200 | plateau 판단 step 수 |
| `plateau_entropy_scale` | 0.3 | plateau 시 entropy scale |
| `plateau_uncovered_scale` | 0.3 | plateau 시 uncovered reward scale |
| `print_interval` | 20 | 로그 출력 및 중간 NPZ 저장 간격 |

주의: 코드 변수명은 `episodes`지만 현재 구현에는 환경 reset과 terminal episode가 없습니다. 실제 의미는 한 run에서 수행하는 연속 action step 수에 가깝습니다.

## 5. 시나리오 데이터와 환경 생성

시나리오는 `data/scenarios/yeongjong_final.json`에서 로드합니다.

JSON에는 OSM 분석에서 얻은 위치 목록과 빈도 정보가 포함되어 있지만, 현재 환경 생성에는 개별 위도·경도나 실제 도로 polyline을 사용하지 않습니다. 실제 계산에 사용하는 필드는 다음 네 개입니다.

```text
id
topology
road_width
num_lanes
```

### 5.1 현재 선택 시나리오

| scenario | topology | road width | lanes | source count | 의미 |
|---:|---|---:|---:|---:|---|
| 0 | STRAIGHT | 7.0 m | 2 | 336 | 좁은 직선 도로 |
| 1 | T_JUNCTION | 7.0 m | 2 | 63 | 좁은 T자 교차로 |
| 2 | CROSS | 7.0 m | 2 | 18 | 좁은 십자 교차로 |
| 3 | STRAIGHT | 21.0 m | 6 | 14 | 넓은 직선 도로 |
| 6 | CROSS | 21.0 m | 6 | 3 | 넓은 십자 교차로 |
| 9 | T_JUNCTION | 28.0 m | 8 | 1 | 매우 넓은 T자 교차로 |

### 5.2 공통 지도 상수

`src.environment.MapGeometry`가 사용하는 상수는 다음과 같습니다.

| 항목 | 값 |
|---|---:|
| map length | 100 m |
| map width | 100 m |
| XY grid size | 0.5 m |
| sidewalk width | 5.0 m |
| median width | 0.5 m |
| building height | 15.0 m |
| 기본 도로 중심 | y = 50 m |
| CROSS/T_JUNCTION 세로 도로 중심 | x = 50 m |

### 5.3 topology별 도로 생성

#### STRAIGHT

- y=50 m를 중심으로 가로 도로를 생성합니다.
- 도로 폭은 scenario의 `road_width`를 사용합니다.
- 중앙부의 median 영역은 일반 도로 target에서 제외합니다.
- 도로 위·아래의 sidewalk 외부 영역을 건물로 채웁니다.

#### CROSS

- y=50 m 가로 도로와 x=50 m 세로 도로를 결합합니다.
- 네 모서리의 도로/인도 외부 영역을 건물로 만듭니다.
- 현재 mesh는 가로·세로 median을 교차로 중앙까지 연속으로 그립니다.

#### T_JUNCTION

- y=50 m 가로 도로는 전체 폭으로 생성합니다.
- x=50 m 세로 도로는 지도 아래쪽에서 교차로 중심까지만 생성합니다.
- 위쪽 전체와 아래쪽 양 모서리를 건물 영역으로 만듭니다.

### 5.4 도로 target 생성

100 m x 100 m 지도를 0.5 m cell center로 순회합니다.

```text
x = (grid_x + 0.5) × 0.5
y = (grid_y + 0.5) × 0.5
```

`MapGeometry.is_road(x, y)`가 참인 XY에 세 개의 높이 slice를 추가합니다.

| scenario | target 수 |
|---:|---:|
| 0 | 7,200 |
| 1 | 10,632 |
| 2 | 13,980 |
| 3 | 24,000 |
| 6 | 43,212 |
| 9 | 44,400 |

모든 target의 objective weight는 동일합니다. 현재 교통량, 차로 중요도, 거리, 객체 종류에 따른 가중치는 없습니다.

### 5.5 설치 가능 영역

설치 가능 여부는 다음 조건을 모두 만족해야 합니다.

```text
건물/negative zone이 아님
도로가 아님
median line이 아님
도로와 평행한 sidewalk 중심선 부근임
```

알고리즘의 정확한 판정 tolerance는 기본적으로 grid size의 절반인 0.25 m입니다. 즉 현재 구현은 5 m sidewalk 전체가 아니라 sidewalk 중심 부근의 좁은 pole line을 설치 가능 영역으로 사용합니다.

MILP/GREEDY는 0.5 m 설치 가능 cell을 순회한 뒤 2 m bin마다 첫 번째 cell만 남깁니다. 각 XY에서 높이 4개를 생성합니다.

| scenario | 2 m XY 후보 | XYZ base 후보 | top-10 orientation 적용 전역 dedup 전 최대 후보 |
|---:|---:|---:|---:|
| 0 | 200 | 800 | 8,000 |
| 1 | 284 | 1,136 | 11,360 |
| 2 | 368 | 1,472 | 14,720 |
| 3 | 100 | 400 | 4,000 |
| 6 | 156 | 624 | 6,240 |
| 9 | 120 | 480 | 4,800 |

마지막 열은 `XYZ base × top 10`입니다. coverage가 완전히 동일한 후보를 제거하므로 실제 MILP 변수 수는 이보다 작을 수 있습니다.

## 6. LiDAR visibility 모델

### 6.1 현재 profile

```yaml
VELODYNE64_R30:
  h_fov_deg: 360.0
  v_fov_deg: 26.9
  max_range_m: 30.0
```

추가 profile로 40 m와 80 m range가 설정되어 있지만 현재 완료 실험은 모두 `VELODYNE64_R30`을 사용했습니다.

### 6.2 방향 좌표 변환

LiDAR에서 target까지의 벡터를 다음과 같이 둡니다.

```text
dx = target_x - lidar_x
dy = target_y - lidar_y
dz = target_z - lidar_z
```

yaw 회전 후 좌표:

```text
dx_y =  dx cos(yaw) + dy sin(yaw)
dy_y = -dx sin(yaw) + dy cos(yaw)
```

pitch 회전 후 좌표:

```text
dx_f =  dx_y cos(pitch) + dz sin(pitch)
dz_f = -dx_y sin(pitch) + dz cos(pitch)
```

azimuth와 elevation:

```text
azimuth   = atan2(dy_y, dx_f)
elevation = atan2(dz_f, sqrt(dx_f^2 + dy_y^2))
distance  = sqrt(dx^2 + dy^2 + dz^2)
```

FOV 조건:

```text
|azimuth|   <= h_fov / 2
|elevation| <= v_fov / 2
distance    <= max_range
```

현재 horizontal FOV는 360도이므로 horizontal 방향은 모두 허용됩니다. 하지만 LiDAR를 pitch로 기울인 방향이 yaw에 따라 달라지므로 yaw가 vertical coverage에는 영향을 줄 수 있습니다.

### 6.3 Open3D raycasting

환경 mesh에는 다음 geometry가 포함됩니다.

- 바닥
- 도로 surface
- median
- lane marking
- sidewalk
- 15 m 높이 건물 block

각 설치 가능 0.5 m XY cell과 각 z 후보에서 모든 target 방향으로 ray를 발사합니다.

```text
origin    = [lidar_x, lidar_y, lidar_z]
direction = (target - origin) / ||target - origin||
```

target까지의 거리를 `d`, Open3D가 반환한 첫 충돌 거리를 `t_hit`라고 할 때 line-of-sight visible 조건은 다음과 같습니다.

```text
visible_los = no finite hit
              OR t_hit >= d - 1e-3
```

여기에 `d <= max_range`를 함께 적용합니다.

raycasting 단계는 장애물과 range만 계산합니다. yaw/pitch FOV는 후보 pose를 평가할 때 별도로 적용합니다.

### 6.4 Visibility cache

cache 파일은 다음 형식입니다.

```text
data/cache/visibility_s{scenario}_{profile}_{hash}.pkl
```

주요 내용:

```text
table[(grid_x, grid_y, z_index)] = packed visible target bit mask
counts[(grid_x, grid_y, z_index)] = visible target 수
target_points
target_weights
z values
scenario/profile metadata
config hash
```

target mask는 `numpy.packbits()`로 압축하여 저장합니다.

GPU Open3D raycasting을 먼저 시도하고 사용할 수 없으면 CPU로 fallback합니다. 현재 검증 환경에서는 Open3D CUDA device가 적합하지 않아 CPU fallback이 사용되었습니다.

## 7. 공통 coverage 계산

LiDAR i가 target j를 관측하면 다음 binary visibility를 정의합니다.

```text
a_ij = 1 if target j is visible from pose i
       0 otherwise
```

K개 pose에 대한 target별 관측 밀도:

```text
d_j = sum_i a_ij
```

현재 `road_detection=1`이므로 covered indicator는 다음과 같습니다.

```text
c_j = 1 if d_j >= 1
      0 otherwise
```

최종 coverage:

```text
covered_count  = sum_j c_j
coverage_ratio = covered_count / number_of_targets
coverage_pct   = 100 × coverage_ratio
```

겹치는 영역은 `d_j`에는 누적되지만 `c_j`는 binary이므로 최종 coverage에는 한 번만 포함됩니다.

MILP/GREEDY는 candidate 생성 시 만든 동일 mask의 union으로 coverage를 계산합니다. MAPPO는 학습 중 동일 cache/FOV 계산을 사용하고, 저장 직전에 `evaluate_coverage()`로 다시 평가합니다.

## 8. MILP

구현 파일은 `src/methods/milp.py`입니다.

### 8.1 Candidate 생성

1. 공통 환경에서 설치 가능 XY를 찾습니다.
2. 2 m bin마다 첫 위치를 선택합니다.
3. z={3,4,5,6}을 결합합니다.
4. 위치별 허용 yaw와 pitch를 생성합니다.
5. 각 pose의 공통 coverage mask를 계산합니다.
6. 단일 센서 coverage 기준 상위 10개 orientation을 남깁니다.
7. coverage mask가 완전히 같은 후보를 deduplication합니다.

### 8.2 Maximum coverage formulation

변수:

```text
x_i ∈ {0,1}: candidate i 선택 여부
y_j ∈ {0,1}: target j covered 여부
```

목적함수:

```text
maximize sum_j y_j
```

coverage constraint:

```text
y_j <= sum_i a_ij x_i,  for every target j
```

budget constraint:

```text
sum_i x_i <= K
```

현재 문제는 exact-K가 아니라 at-most-K입니다. 추가 LiDAR가 새로운 target을 덮지 않으면 K보다 적게 선택할 수 있습니다.

### 8.3 Solver 구현

- 기본 backend는 Gurobi입니다.
- candidate-target incidence matrix를 SciPy sparse matrix로 구성합니다.
- Gurobi `MVar`와 `addMConstr`를 사용합니다.
- GREEDY solution을 warm start로 제공합니다.
- Gurobi 관련 예외가 발생하면 PuLP/CBC fallback을 시도합니다.

K=1은 모든 단일 후보 중 coverage 최대 후보를 GREEDY가 정확하게 찾을 수 있으므로 solver를 생략합니다.

이전 budget에서 이미 100% coverage를 달성했다면 더 큰 K의 solver도 생략합니다.

### 8.4 Budget frontier

각 K에서 다음 후보 중 coverage가 가장 높은 결과를 저장합니다.

```text
현재 K의 GREEDY warm start
현재 K의 MILP raw solution
이전 K까지 저장한 best feasible solution
```

이 때문에 `frontier_coverage_pct`는 K가 증가할 때 감소하지 않습니다.

MILP JSON에는 다음 두 coverage가 구분되어 있습니다.

```text
raw_coverage_pct       현재 solver가 반환한 raw solution
frontier_coverage_pct  현재 K 이하에서 얻은 최고 feasible solution
```

기본 비교 CSV는 `frontier_coverage_pct`를 MILP coverage로 사용합니다.

## 9. GREEDY

구현 파일은 `src/methods/greedy.py`입니다.

GREEDY는 MILP의 `build_oriented_candidates()`를 직접 import하므로 MILP와 동일한 pruned candidate set을 사용합니다.

초기 상태:

```text
C = empty covered target set
S = empty selected candidate set
```

각 step에서 후보 i의 marginal gain을 계산합니다.

```text
gain(i) = |A_i minus C|
```

가장 큰 gain을 가진 후보를 선택합니다.

```text
i* = argmax_i gain(i)
S  = S union {i*}
C  = C union A_i*
```

다음 조건 중 하나를 만족할 때 종료합니다.

```text
|S| = K
or best marginal gain <= 0
```

동률이면 총 단일 coverage가 큰 후보를 우선하고, 그래도 같으면 candidate ID가 작은 후보를 선택합니다. 학습과 random seed가 없는 deterministic baseline입니다.

## 10. MAPPO

구현 파일은 `src/methods/mappo.py`입니다. legacy MAPPO 구조를 현재 공통 환경에 맞게 native port한 코드입니다.

### 10.1 Agent와 공유 정책

- LiDAR 수 K만큼 agent를 생성합니다.
- 모든 agent는 하나의 actor policy를 공유합니다.
- critic은 모든 agent state를 연결한 global state를 입력으로 받습니다.
- K가 달라지면 critic input dimension도 달라집니다.

### 10.2 Spawn 초기화

1. 공통 설치 가능 0.5 m grid를 찾습니다.
2. MILP와 같은 2 m bin 방식으로 subsampling합니다.
3. 각 spawn의 높이별 raw visibility count 중 최대값으로 순위를 정합니다.
4. 상위 200개 spawn을 warm-start 후보로 둡니다.
5. 첫 spawn은 seed 기반 random으로 고르고 이후 farthest-point 방식으로 agent를 분산합니다.

초기 높이는 z 후보의 중앙 index이며 현재는 5 m입니다. 초기 yaw와 pitch는 허용 후보에서 seed 기반으로 선택합니다.

### 10.3 State

agent 하나의 state dimension은 다음과 같습니다.

```text
5 pose features + 7×7 local map + 7 global hints
= 5 + 49 + 7
= 61
```

Pose feature:

```text
x normalized by map grid width
y normalized by map grid height
z index normalized
road normal 기준 relative yaw / 180
pitch normalized to allowed range
```

Local map:

- agent 주변 7×7 위치의 raw visibility count를 사용합니다.
- 각 local cell은 grid index 2칸, 즉 1 m 간격으로 sampling합니다.
- scenario 내 최대 visibility count로 정규화합니다.

Global hint 7개:

```text
nearest road direction vector x, y
nearest road distance normalized
heading-road alignment
uncovered centroid direction x, y
uncovered centroid distance normalized
```

### 10.4 Action

action dimension은 11입니다.

| action | 의미 | 변화량 |
|---:|---|---:|
| 0 | stay | 없음 |
| 1 | x+ | +2 m |
| 2 | x- | -2 m |
| 3 | y+ | +2 m |
| 4 | y- | -2 m |
| 5 | yaw+ | +5 deg |
| 6 | yaw- | -5 deg |
| 7 | pitch+ | +2.5 deg |
| 8 | pitch- | -2.5 deg |
| 9 | z+ | +1 m |
| 10 | z- | -1 m |

action mask는 다음 행동을 막습니다.

- 지도 밖으로 나가는 XY 이동
- 설치 불가능 영역으로 이동
- 허용 relative yaw 범위 초과
- pitch 범위 초과
- z 범위 초과

### 10.5 Actor-Critic network

Actor local map encoder:

```text
Conv2d(1,16,3,padding=1)
ReLU
Conv2d(16,32,3,padding=1)
ReLU
Flatten
```

Actor head:

```text
[pose + hint + CNN feature]
→ Linear(256)
→ ReLU
→ Linear(128)
→ ReLU
→ Linear(11 action logits)
```

Centralized critic:

```text
concatenated state of K agents
→ Linear(256)
→ ReLU
→ Linear(128)
→ ReLU
→ Linear(1 value)
```

### 10.6 Reward

target density를 `density_j`라고 할 때 score map은 다음과 같습니다.

```text
score_j = min(density_j / road_detection, 1)
```

모든 target weight가 0.01이므로 현재 score는 covered target 수에 0.01을 곱한 값과 같습니다.

Team reward:

```text
delta_score        = current_score - previous_score
delta_reward       = 25 × delta_score
absolute_reward    = 0.1 × current_score
overlap_penalty    = w_overlap_pen × overlap_score

team_reward = delta_reward + absolute_reward - overlap_penalty
```

현재 `w_overlap_pen=0`이므로 explicit overlap penalty는 비활성화되어 있습니다. 다만 coverage score가 target별로 1에서 포화되므로 중복 관측 자체는 추가 coverage 보상을 만들지 않습니다.

Agent별 shaping reward:

- 다른 agent와 멀리 떨어지는 보상
- 5 m 미만으로 가까울 때 penalty
- 가장 가까운 미커버 target 쪽으로 이동한 거리 보상
- 지도 경계에서 바깥쪽을 향할 때 penalty
- 해당 agent만 제공하는 unique coverage의 marginal gain 보상

최종 agent reward는 다음과 같습니다.

```text
reward_i = team_reward + individual_reward_i
```

buffer 저장 전 reward에 0.1을 곱하고 [-10, 10]으로 clipping합니다.

### 10.7 PPO update 의도

코드가 의도한 PPO는 다음 요소를 포함합니다.

- old policy sampling
- clipped surrogate objective
- centralized value loss
- clipped value update
- GAE
- entropy bonus와 decay
- gradient norm clipping 1.0
- minibatch update

그러나 현재 old/new log probability shape와 multi-agent GAE trajectory 구성에 오류가 있습니다. 현재 결과를 표준 MAPPO 결과로 해석하면 안 됩니다. 수정 필요 사항은 문서 뒤쪽에 구체적으로 기록했습니다.

### 10.8 Best pose 저장

학습 중 coverage가 기존 best보다 높아지면 NPY를 저장합니다. 100 step마다 5번 deterministic action을 수행한 평가 pose도 비교합니다.

학습 종료 후 agent별 yaw를 허용 yaw 후보로 바꿔보는 coordinate-wise yaw tuning을 수행합니다. 더 높은 coverage가 나오면 이를 저장합니다.

마지막으로 저장 pose를 공통 `evaluate_coverage()`로 다시 평가하고 JSON을 생성합니다.

## 11. 결과 파일 형식

### 11.1 Pose NPY

각 행은 LiDAR 한 대입니다.

```text
[x, y, z, yaw, pitch]
```

dtype은 `float32`입니다.

### 11.2 공통 JSON schema

```text
method
scenario_id
topology
lidar_profile
lidar_model
max_range_m
budget_k
budget_mode
candidate_mode
action_space
num_selected
positions_npy
coverage_pct
coverage_ratio
covered_count
num_targets
episodes
seed
trial
mappo_tag
config_hash
cache_key
created_at
```

### 11.3 MILP 추가 필드

```text
raw_coverage_pct
frontier_coverage_pct
frontier_source_budget_k
frontier_source_method
solver_status
solver_backend
solver_error
mip_gap
extra.selected_candidate_ids
extra.frontier_selected_candidate_ids
```

### 11.4 MAPPO 추가 필드

```text
train_log_npz
extra.trainer
extra.legacy_action_dim
extra.best_train_coverage_pct
extra.best_eval_coverage_pct
extra.best_internal_saved_coverage_pct
extra.best_internal_saved_score
```

### 11.5 MAPPO train log NPZ

```text
episode
coverage
coverage_pct
score
reward
overlap_bonus
team_reward
delta_score_reward
abs_cov_reward
overlap_penalty
dist_bonus
dist_penalty
uncovered_bonus
outward_penalty
marginal_gain
indiv_reward_mean
eval_episode
eval_coverage
eval_score
```

### 11.6 비교 CSV

핵심 결과 파일:

```text
results/journal_r30_top10_20260707/tables/compare_methods.csv
```

CSV method 구성:

```text
MILP_BUDGET: 36 rows
GREEDY:      36 rows
MAPPO:      180 rows
MAPPO_MEAN:  36 rows
```

`MAPPO_MEAN`은 scenario/K별 5개 trial의 평균입니다. 현재 `std_coverage_pct`는 모집단 표준편차, 즉 `ddof=0`입니다.

## 12. 현재 완료 결과

### 12.1 결과 파일 수

```text
GREEDY: 36 JSON + 36 NPY
MILP:   36 JSON + 36 NPY
MAPPO: 180 JSON + 180 NPY + 180 train-log NPZ
CSV:     1
PNG:   943
PDF:     7
```

### 12.2 MILP coverage

단위는 `%`입니다. 저장된 frontier coverage를 표시합니다.

| scenario | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58.403 | 98.667 | 99.875 | 100.000 | 100.000 | 100.000 |
| 1 | 60.092 | 79.628 | 90.613 | 99.925 | 100.000 | 100.000 |
| 2 | 55.086 | 71.645 | 79.378 | 87.110 | 93.555 | 100.000 |
| 3 | 52.175 | 95.508 | 99.767 | 99.963 | 100.000 | 100.000 |
| 6 | 46.304 | 73.250 | 87.871 | 99.153 | 99.153 | 99.993 |
| 9 | 51.045 | 78.872 | 88.649 | 96.124 | 98.230 | 99.185 |

### 12.3 GREEDY coverage

| scenario | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58.403 | 98.667 | 99.875 | 100.000 | 100.000 | 100.000 |
| 1 | 60.092 | 79.628 | 90.613 | 99.925 | 100.000 | 100.000 |
| 2 | 55.086 | 71.645 | 79.378 | 87.110 | 93.555 | 100.000 |
| 3 | 52.175 | 95.508 | 99.767 | 99.963 | 100.000 | 100.000 |
| 6 | 46.304 | 73.250 | 81.938 | 89.052 | 95.131 | 99.993 |
| 9 | 51.045 | 78.872 | 88.649 | 96.124 | 98.230 | 99.185 |

### 12.4 MAPPO coverage

각 값은 현재 CSV에 저장된 `5 trials mean +/- population standard deviation`입니다.

| scenario | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58.369 +/- 0.027 | 99.331 +/- 0.327 | 99.917 +/- 0.053 | 100.000 +/- 0.000 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| 1 | 56.159 +/- 2.852 | 78.057 +/- 1.119 | 97.568 +/- 0.774 | 99.221 +/- 0.759 | 99.919 +/- 0.048 | 99.985 +/- 0.030 |
| 2 | 54.964 +/- 0.000 | 70.951 +/- 0.414 | 83.259 +/- 0.586 | 95.918 +/- 2.128 | 97.847 +/- 1.969 | 99.246 +/- 0.368 |
| 3 | 51.925 +/- 0.204 | 96.228 +/- 0.318 | 99.186 +/- 0.413 | 99.897 +/- 0.094 | 100.000 +/- 0.000 | 100.000 +/- 0.000 |
| 6 | 46.107 +/- 0.029 | 74.701 +/- 0.483 | 86.399 +/- 0.402 | 95.890 +/- 0.976 | 98.821 +/- 0.223 | 99.323 +/- 0.245 |
| 9 | 40.469 +/- 8.268 | 78.226 +/- 1.281 | 88.460 +/- 0.882 | 93.295 +/- 1.239 | 96.757 +/- 0.473 | 97.891 +/- 0.471 |

### 12.5 전체 조합 기술 통계

36개 scenario/K 조합의 단순 산술평균입니다. 서로 다른 target 수와 난이도의 문제를 동일 가중치로 평균한 값이므로 논문의 주 결론보다 보조 기술통계로 사용해야 합니다.

| method | 36개 조합 평균 coverage |
|---|---:|
| MILP_BUDGET | 87.200% |
| MAPPO_MEAN | 87.063% |
| GREEDY | 86.643% |

MAPPO 평균이 MILP frontier보다 높은 조합은 36개 중 8개입니다. 이는 MAPPO가 MILP보다 이론적으로 우월하다는 의미가 아닙니다. 현재 두 방법의 orientation feasible space가 다르고 MILP에도 time/gap 제한이 있기 때문입니다.

### 12.6 MILP solver 상태

| solver status | 개수 | 해석 |
|---|---:|---|
| `2` | 16 | Gurobi가 설정된 tolerance 내에서 종료 |
| `9` | 6 | 300초 time limit에서 feasible solution 저장 |
| `SKIPPED_EXACT_K1` | 6 | pruned candidate set의 K=1을 analytic하게 선택 |
| `SKIPPED_PROVEN_FULL_COVERAGE` | 8 | 이전 frontier가 이미 100%라 solver 생략 |

status 2인 16개도 저장된 relative gap이 모두 0보다 크며 최대 약 4.70%입니다.

time-limit 6개:

```text
scenario 1: K=2, K=3
scenario 2: K=2, K=3, K=4, K=5
```

이 6개는 `mip_gap=inf`로 저장되어 있어 optimality proof가 없습니다.

## 13. 결과 그래프와 3D 확인

### 13.1 비교 그래프

```text
results/journal_r30_top10_20260707/figures/comparison/
```

파일:

```text
coverage_compare_s0.png / pdf
coverage_compare_s1.png / pdf
coverage_compare_s2.png / pdf
coverage_compare_s3.png / pdf
coverage_compare_s6.png / pdf
coverage_compare_s9.png / pdf
coverage_compare_all_scenarios.png / pdf
```

그래프는 `MILP_BUDGET`, `GREEDY`, `MAPPO_MEAN`을 비교하고 MAPPO trial 표준편차 band를 표시합니다.

현재 이 comparison figure를 생성한 전용 함수는 source에 남아 있지 않고 결과 파일만 존재합니다. 논문 재현성을 위해 figure 생성 코드를 정식 CLI에 포함해야 합니다.

### 13.2 저장 figure 재생성

```bash
.venv/bin/python main.py visualize \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --methods milp,greedy,mappo \
  --mappo-tag journal_r30_top10_20260707 \
  --results-root results/journal_r30_top10_20260707 \
  --output-dir results/journal_r30_top10_20260707/figures \
  --output-3d-dir results/journal_r30_top10_20260707/3d \
  --save-3d
```

각 결과에서 다음 파일을 생성합니다.

```text
topview PNG
coverage heatmap PNG
optional 3D Matplotlib snapshot PNG
MAPPO training graph PNG
```

### 13.3 대화형 Matplotlib viewer

```bash
.venv/bin/python view_result.py \
  --config configs/experiment.yaml \
  --results-root results/journal_r30_top10_20260707 \
  --method mappo \
  --scenario-id 0 \
  --k 1 \
  --mappo-tag journal_r30_top10_20260707 \
  --viewer matplotlib
```

마우스와 toolbar로 회전, 이동, 확대할 수 있습니다.

### 13.4 Open3D viewer

```bash
.venv/bin/python view_result.py \
  --config configs/experiment.yaml \
  --results-root results/journal_r30_top10_20260707 \
  --method milp \
  --scenario-id 2 \
  --k 4 \
  --viewer open3d
```

### 13.5 특정 JSON 바로 열기

```bash
.venv/bin/python view_result.py \
  --config configs/experiment.yaml \
  --result-json results/journal_r30_top10_20260707/milp/milp_velodyne64_r30_journal_r30_top10_20260707_id2_k4.json \
  --viewer matplotlib
```

## 14. 실행 방법

### 14.1 주의

- `main.py full`은 현재 실제 전체 실험을 실행하지 않는 scaffold입니다.
- 전체 실험은 subcommand를 순서대로 실행해야 합니다.
- MAPPO 구현 오류를 고치기 전에는 새 official MAPPO 결과를 만들지 않는 것이 좋습니다.
- 같은 MAPPO tag로 재실행하면 기존 NPY best가 새 run에 섞일 수 있으므로 수정 전에는 같은 tag 재사용을 피해야 합니다.

### 14.2 Dry-run

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

Dry-run은 candidate 계획과 출력 경로만 확인하며 학습, solve, 파일 저장을 수행하지 않습니다.

### 14.3 새 실험 실행 형식

아래는 명령 형식입니다. MAPPO 수정 및 검증이 완료된 뒤 새 tag와 새 결과 폴더로 실행해야 합니다.

```bash
RESULTS_ROOT=results/journal_new
EXPERIMENT_TAG=journal_new

.venv/bin/python main.py greedy \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --tag "$EXPERIMENT_TAG" \
  --output-root "$RESULTS_ROOT"

.venv/bin/python main.py milp \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --tag "$EXPERIMENT_TAG" \
  --output-root "$RESULTS_ROOT"

.venv/bin/python main.py mappo \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --episodes 20000 \
  --trials 5 \
  --tag "$EXPERIMENT_TAG" \
  --output-root "$RESULTS_ROOT"

.venv/bin/python main.py compare \
  --config configs/experiment.yaml \
  --scenario-ids 0,1,2,3,6,9 \
  --k-values 1,2,3,4,5,6 \
  --mappo-tag "$EXPERIMENT_TAG" \
  --output-root "$RESULTS_ROOT"
```

현재 `compare` 구현은 `--scenario-ids`와 `--k-values`를 실제 row filtering에 적용하지 않는 문제가 있으므로, 하나의 결과 root에는 한 official experiment만 저장하는 것이 안전합니다.

### 14.4 test.py runner

`test.py`는 `--execute`가 없으면 실행 계획만 출력합니다.

```bash
.venv/bin/python test.py plan \
  --config configs/experiment.yaml \
  --scenario-ids 0 \
  --k-values 1 \
  --episodes 10 \
  --trials 1
```

실제 실행에는 `--execute`가 필요합니다.

```bash
.venv/bin/python test.py greedy \
  --config configs/experiment.yaml \
  --scenario-ids 0 \
  --k-values 1 \
  --tag smoke \
  --output-root results/smoke \
  --execute
```

## 15. 검증 결과

2026-08-07 기준 코드와 완료 결과를 다음 방식으로 검토했습니다.

### 15.1 문법과 경량 테스트

```text
py_compile: 통과
tests/test_pipeline.py: 3 tests 통과
MILP/GREEDY/MAPPO dry-run: 통과
```

테스트 항목:

- scenario별 cache key 분리
- LiDAR range profile 로드
- MAPPO plan에 scenario/K 반영
- 결과 schema 기본 필드
- Open3D raycast cache 생성
- coverage 범위 0~100%

### 15.2 Toy MILP 검증

작은 binary coverage 문제를 brute force와 Gurobi sparse MILP로 비교했습니다.

```text
K=1: Gurobi coverage = brute-force optimum
K=2: Gurobi coverage = brute-force optimum
K=3: Gurobi coverage = brute-force optimum
```

따라서 현재 sparse matrix MILP constraint 방향과 objective 구성은 정상입니다.

### 15.3 완료 결과 252개 재평가

모든 JSON/NPY를 다시 로드해 공통 coverage mask로 재계산했습니다.

```text
GREEDY JSON: 36
MILP JSON:   36
MAPPO JSON: 180
총 검사:    252
```

검사 결과:

```text
기록 coverage와 재계산 coverage 일치
covered_count 일치
num_targets 일치
cache_key 일치
config_hash 일치
num_selected <= K
설치 가능 XY 조건 통과
z grid 조건 통과
yaw 허용 범위 통과
pitch 범위와 2.5도 grid 통과
동일 run 내부 완전 중복 pose 없음
MILP/GREEDY frontier monotonicity 통과
문제 발견 수: 0
```

### 15.4 MAPPO 학습 로그 검사

```text
180개 train log 모두 episode array 길이 20,000
180개 train log 모두 eval record 200개
NaN/Inf 없음
MAPPO 내부 coverage와 공통 evaluator sample 비교 일치
```

이는 저장 파일과 coverage 평가가 정상이라는 의미입니다. PPO gradient 계산이 올바르다는 의미는 아닙니다.

## 16. 논문 사용 전 필수 수정사항

### 16.1 P0: MAPPO PPO ratio shape 오류

현재 old log probability는 buffer stacking 후 `[B,1]`, 새 log probability는 `[B]`입니다.

```text
new_logp shape = [B]
old_logp shape = [B,1]
new_logp - old_logp shape = [B,B]
```

따라서 transition별 PPO ratio가 아니라 batch 내 모든 transition의 pairwise matrix로 surrogate loss가 계산됩니다.

필요 수정:

```text
log probability를 buffer에 scalar로 저장
또는 stack 후 squeeze(-1)
ratio shape가 반드시 [B]인지 assertion 추가
```

현재 MAPPO 결과는 이 문제를 수정한 뒤 전부 재학습해야 합니다.

### 16.2 P0: Multi-agent GAE trajectory 오류

현재 shared buffer에는 같은 timestep의 agent transition들이 연속으로 들어갑니다.

```text
t0-agent0, t0-agent1, ..., t0-agentK,
t1-agent0, t1-agent1, ...
```

GAE는 이 배열을 하나의 temporal trajectory로 처리하므로 agent0의 next transition이 agent1로 연결됩니다. 또한 update chunk 끝의 next value를 0으로 두면서 `done=False` 상태와 모순됩니다.

필요 수정:

- buffer에 `[time, agent, ...]` 구조를 유지
- timestep별 reward/value/next value를 사용
- terminal/reset 정의 또는 continuing-task bootstrap 구현
- agent별 advantage 또는 team trajectory 설계 명시
- `episodes`와 `steps` 용어를 실제 loop와 일치시킴

### 16.3 P1: 알고리즘 feasible space 통일

현재:

```text
MILP/GREEDY = 위치별 top-10 orientation
MAPPO       = 전체 yaw/pitch action 범위
```

선택 가능한 해결 방법:

1. 세 방법 모두 동일한 discrete candidate set을 사용합니다.
2. MILP/GREEDY의 topM을 제거해 전체 orientation을 사용합니다.
3. 계산량 때문에 topM이 필요하면 MILP를 global optimum이 아니라 `pruned-candidate MILP baseline`으로 명시합니다.
4. topM={5,10,20,all} sensitivity analysis를 추가합니다.

### 16.4 P1: MILP optimality 보고

논문 표에는 최소한 다음을 함께 보고해야 합니다.

```text
solver status
incumbent objective
best bound
relative MIP gap
runtime
candidate count
constraint/edge count
```

현재 JSON에는 runtime과 best bound가 저장되지 않으므로 결과 schema를 확장해야 합니다.

time limit 또는 `gap=inf` 결과를 optimum이라고 표현하면 안 됩니다.

### 16.5 P1: 통계 처리

현재 MAPPO 표준편차는 모집단 표준편차 `ddof=0`입니다.

논문용 권장 항목:

- sample standard deviation `ddof=1`
- 95% confidence interval
- trial별 raw point 표시
- paired seed 설계
- Wilcoxon signed-rank 또는 조건에 맞는 통계검정
- effect size
- seed와 software/hardware 환경 기록

### 16.6 P1: Cache 재현성

현재 cache hash에는 `MapGeometry` 상수와 환경 코드 버전이 포함되지 않습니다. 환경 코드를 바꾸고 cache model 문자열을 갱신하지 않으면 stale cache가 재사용될 수 있습니다.

필요 수정:

- geometry constants를 config/hash에 포함
- visibility algorithm version을 명시적으로 증가
- source revision 또는 schema version 기록
- cache 생성 시 atomic write 사용
- cache validation test 강화

### 16.7 P1: 결과 혼합 방지

현재 compare의 scenario/K filter가 실제 수집 결과에 적용되지 않습니다. MILP/GREEDY JSON에는 experiment tag 전용 필드가 없어 같은 root의 여러 run이 섞일 수 있습니다.

필요 수정:

- 모든 method JSON에 `experiment_tag` 저장
- compare에서 profile/scenario/K/tag/config_hash를 모두 filtering
- 조합별 중복 결과 검출
- 기대 결과 수 미달 시 실패 처리

### 16.8 P1: MAPPO 기존 결과 오염 방지

같은 output tag의 NPY가 이미 존재하면 MAPPO가 이를 global best로 불러옵니다. 설정이나 코드가 바뀐 새 run에서도 예전 pose가 남을 수 있습니다.

필요 수정:

- resume를 명시적 옵션으로 분리
- config hash, seed, episode가 일치할 때만 resume
- 새 run은 기존 파일이 있으면 실패하거나 새 directory 사용

### 16.9 P2: 물리 모델 범위 명시

현재 visibility는 idealized geometric coverage입니다.

포함하지 않는 요소:

- Velodyne 64개 channel별 실제 elevation angle
- horizontal/vertical angular resolution
- 회전 주기와 point density
- material reflectivity
- incidence angle
- 비, 안개, 눈 등 기상
- 차량·보행자에 의한 동적 occlusion
- detection algorithm의 range별 성능
- calibration error와 pole vibration

논문에서는 이를 `idealized line-of-sight volumetric coverage`로 명확히 정의해야 합니다. 실제 detection probability로 표현하면 안 됩니다.

### 16.10 P2: 교차로와 설치 영역 정의

- CROSS/T_JUNCTION 중앙부에 median target이 일부 포함됩니다.
- mesh의 median은 교차로 중앙까지 연속으로 그려집니다.
- 설치 가능 영역은 sidewalk 전체가 아니라 중심선 부근입니다.
- viewer에서 보이는 설치 가능 영역 tolerance가 알고리즘 판정보다 넓을 수 있습니다.

이 정의를 논문 그림과 수식에 그대로 명시하거나 geometry를 수정한 뒤 cache와 결과를 다시 생성해야 합니다.

### 16.11 P2: Viewer 통계

대화형 viewer의 global coverage는 공통 evaluator를 사용하므로 정확합니다. 그러나 agent별 visible/unique/overlap과 beam은 raycasting occlusion 없이 거리와 FOV만 계산합니다.

viewer 통계를 논문 수치로 사용하지 말고, 공통 `coverage_mask_for_pose()`를 사용하도록 수정해야 합니다.

### 16.12 P2: 프로젝트 재현성

현재 저장소에는 dependency lock 파일이 없습니다.

필요 항목:

```text
pyproject.toml 또는 requirements.txt
Python version
NumPy/SciPy/PyTorch/Open3D/Gurobi/PuLP/pandas/matplotlib/PyYAML version
CUDA/driver version
CPU/GPU/RAM 정보
Gurobi version과 academic license 조건
실험 시작/종료 timestamp
Git revision 또는 source archive hash
```

또한 `main.py`와 `test.py`의 중복 runner logic, no-op인 `main.py full`, source에 없는 comparison figure 생성 절차를 정리해야 합니다.

## 17. 논문에 보고해야 할 최소 실험 정보

### Environment

```text
100 m × 100 m parametric road map
0.5 m target/raycast grid
topology: STRAIGHT, CROSS, T_JUNCTION
road width and lane count by scenario
5 m sidewalk, 0.5 m median, 15 m buildings
target heights: 1.0, 1.5, 2.0 m
```

### LiDAR

```text
profile name
360 deg horizontal FOV
26.9 deg vertical FOV model
30 m maximum range
pose variables and absence of roll
installation height and orientation constraints
```

### Candidate/action domain

```text
XY step
Z values
yaw relative range and step
pitch range and step
topM pruning and dedup rule
at-most-K budget definition
```

### Visibility and objective

```text
Open3D raycasting
mesh obstacle definition
line-of-sight tolerance 1e-3 m
range/FOV equations
binary union coverage
road_detection threshold
target weighting
```

### Algorithms

```text
MILP formulation and solver settings/status/gap
GREEDY marginal gain and tie break
MAPPO state/action/network/reward/PPO parameters
trial count, seed scheme, training step count
best-pose selection and post-processing
```

### Statistics

```text
trial raw values
mean, sample standard deviation, confidence interval
paired comparison method
solver uncertainty and time-limit cases
```

### Reproducibility

```text
source revision
config hash
cache version
software/hardware versions
result directory/tag
exact command line
```

## 18. 의존성

현재 코드에서 사용하는 주요 Python package는 다음과 같습니다.

```text
numpy
scipy
torch
open3d
gurobipy
pulp
pandas
matplotlib
PyYAML
```

현재 workspace는 `.venv`의 Python 3.12를 사용합니다. dependency manifest가 없으므로 새 시스템에서 동일 환경을 재현하려면 버전 고정 파일을 먼저 만들어야 합니다.

## 19. 검증 명령

문법 검사:

```bash
.venv/bin/python -m py_compile \
  main.py test.py view_result.py \
  src/environment.py src/outputs.py src/visualize.py \
  src/methods/milp.py src/methods/greedy.py src/methods/mappo.py \
  tests/test_pipeline.py
```

경량 파이프라인 테스트:

```bash
.venv/bin/python tests/test_pipeline.py
```

결과 개수 확인:

```bash
find results/journal_r30_top10_20260707/greedy -name '*.json' | wc -l
find results/journal_r30_top10_20260707/milp -name '*.json' | wc -l
find results/journal_r30_top10_20260707/mappo -name '*.json' | wc -l
find results/journal_r30_top10_20260707/mappo -name '*_train_log.npz' | wc -l
```

비교 CSV 확인:

```bash
.venv/bin/python - <<'PY'
import pandas as pd

path = 'results/journal_r30_top10_20260707/tables/compare_methods.csv'
df = pd.read_csv(path)
print(df.groupby('method').size())
print(df[df['method'].isin(['MILP_BUDGET', 'GREEDY', 'MAPPO_MEAN'])])
PY
```

## 20. 현재 상태 요약

```text
환경/target/raycast cache: 구현 및 결과 생성 완료
공통 coverage evaluator: 정상
겹침 binary union 처리: 정상
MILP formulation: toy brute-force 검증 통과
GREEDY: MILP와 동일 candidate set 사용
저장된 252개 결과 coverage 무결성: 통과
MAPPO 내부/최종 coverage 일치: 통과
MAPPO PPO optimization: 수정 필수
MILP exact optimality: 일부 조건에서 미보장
비교 통계와 결과 filtering: 보완 필요
논문용 최종 official 결과: MAPPO 수정 후 재생성 필요
```

현재 완료 결과는 코드의 coverage 계산과 파일 저장 파이프라인을 검증하는 데 사용할 수 있습니다. 논문의 최종 성능표에는 MAPPO 구현과 비교 조건을 수정한 뒤 새 output tag로 다시 생성한 결과를 사용해야 합니다.

## 21. License

이 저장소는 오픈소스가 아닙니다. 코드, 설정, 데이터, 문서, 결과물을 논문, 학위논문, 학술대회, 저널, 연구보고서, 상업적 제품, 서비스, 특허, 기술이전 등에 사용하려면 저자의 사전 서면 허가가 필요합니다.

자세한 조건은 `LICENSE`를 확인하세요.
