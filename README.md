# 3D LiDAR Optimal Deployment

> 공통 엔진 리팩터링 1차 구현: Roadside/Robot application CLI와 가중 GREEDY/MILP를
> 추가했습니다. 구조, 재현 명령, 지원 범위는 [REFACTOR_PROGRESS.md](REFACTOR_PROGRESS.md)를
> 확인하세요. 로컬 UI는 `.venv/bin/python placement_ui.py --port 8766`으로 실행합니다.
> 복수 알고리즘/K 비교와 실행별 새 MAPPO 학습을 지원합니다.
> 지도/가중치 조건부 사전학습·전이 RL은 아직 구현되지 않았습니다.

**연구 질문:** 제한된 수의 roadside LiDAR를 어디에, 어떤 높이와 방향으로 배치하면 도로의 관측 가능 영역을 최대화할 수 있는가?

이 프로젝트는 GREEDY, MILP, MAPPO를 공통 도로 geometry와 visibility/coverage 기준으로 비교합니다. 현재는 **정적 파라메트릭 도로의 기하학적 관측 범위 최적화**입니다. 실제 검출 정확도, 동적 차량 가림, 새로운 맵으로의 전이, 도시 전체 최소 센서 수 최적화는 검증한 기능이 아닙니다.

> 문서 확인일: 2026-09-30. 최신 MAPPO 결과: `results/mappo_v3_official_20260928_130911/`.
> 과거 `journal_r30*` 결과는 삭제된 과거 실험이며 현재 결과와 혼용하지 않습니다.
> 생성 결과는 ignored 경로이므로 저장소 clone만으로 포함되지 않습니다.
> 기존 세미나와 `seminar_k11_20260921`은 MAPPO v2 결과입니다. v3는 2026-09-28에 180/180 runs를 완료했습니다. v2와 v3 및 UI 예제의 수치를 구분하세요.

## 배치 Workbench: 현재 구현과 접속

현재 프로젝트에는 기존 Roadside 연구 파이프라인, 공통 가중 최적화 엔진,
Robot/Platform 예제 파이프라인, 로컬 웹 UI가 함께 있습니다.
기존 `milp/greedy/mappo` 명령은 유지되며 새 `roadside/robot` 명령을 추가했습니다.
새로운 지도와 가중치를 조건으로 받는 범용 RL은 아직 구현·학습되지 않았습니다.

### URL이 열리지 않을 때

현재 작업에 사용하는 URL은 **http://127.0.0.1:8766/** 입니다.
UI 서버는 연구 서버의 loopback 주소에서만 수신합니다. 아래 명령은 서버 포트 8766 기준입니다.
`--port`를 생략하면 코드 기본값은 8765이므로 실행 명령과 포트 전달 번호를 맞춰야 합니다.
다른 컴퓨터 브라우저의 `127.0.0.1`은 그 컴퓨터 자체이므로 SSH 연결만으로는 열리지 않습니다.

1. 서버에서 실행 여부를 확인합니다.

```bash
curl -I http://127.0.0.1:8766/
```

2. 서버가 없으면 서버 터미널에서 실행합니다. 이미 200 응답이면 중복 실행하지 않습니다.

```bash
cd /home/robotserver/hj_ws
.venv/bin/python placement_ui.py --port 8766
```

3. **브라우저를 사용하는 PC의 로컬 터미널**에서 아래 명령을 실행하고 켜둡니다.

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 ai-server
```

이후 로컬 브라우저에서 **http://127.0.0.1:8766/** 를 엽니다.
`ai-server` 별칭이 없고 기존 유선 주소를 사용하는 경우:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 robotserver@10.10.10.2
```

로컬 8766이 사용 중이면 `-L 8767:127.0.0.1:8766`으로 변경하고
http://127.0.0.1:8767/ 에 접속합니다. 서버 포트는 그대로입니다.
VS Code Remote SSH에서는 **Ports → Forward a Port → 8766 → Open in Browser**로도 연결할 수 있습니다.
서버에서 HTTP 200이 확인되는데 PC에서 열리지 않으면 포트 전달과 SSH 접속 주소를 확인합니다.
현재 8766 서버 로그는 `/tmp/lidar-placement-ui-8766.log`이며, 직접 실행하면 로그는 해당 터미널에 출력됩니다.

### UI에서 할 수 있는 작업

1. Box 플랫폼, 파라메트릭 도로 또는 사용자 Mesh JSON을 불러옵니다.
2. 설치 후보별 visibility를 계산해 서버 메모리에 보관합니다.
3. K 및 coverage/importance/overlap/count/cost 가중치를 조절합니다.
4. 필요하면 XY 사각형 영역과 중요도 배수를 지정합니다.
5. GREEDY/MILP로 재계산하거나 후보 선택형 MAPPO를 새로 학습하고 3D 장면에서 결과를 확인합니다.
6. 배치·가중치·목적값·관측 밀도가 담긴 JSON을 다운로드합니다.

알고리즘 체크박스로 GREEDY/MILP/MAPPO를 복수 선택하고, 단일 K 또는 시작~종료 K를 지정합니다.
예를 들어 세 알고리즘과 K=3~5를 선택하면 동일 가중치로 9개 조합을 순차 실행합니다.
모든 방법은 설정 변경 후 실행 버튼을 눌러야 합니다. 변경 시 이전 비교 결과는 화면에서 무효화됩니다.
표와 그래프에서 coverage, 중복 관측 target 비율, 미관측 target 비율/수, 가중 목적값을 비교합니다.
중복 비율은 `count(density >= 2) / target_count`이며 보상식의 중복 누적 횟수와는 다릅니다.
미관측은 target 표본 기준이며 연속 사각지대 면적을 계산한 값이 아닙니다.
K는 at-most 예산입니다. 표의 설치 수가 실제 선택 수이며 예산보다 작을 수 있습니다.
표의 지도 선택 버튼이나 그래프 점을 선택하면 해당 알고리즘/K의 3D 배치로 전환합니다.
결과 그래프 선택 영역에서 알고리즘/K별 묶음 또는 표의 개별 그래프 체크박스를 복수 선택할 수 있습니다.
이 선택은 기존 결과의 표시만 바꾸며 재계산하거나 가중치를 변경하지 않습니다.
묶음 막대(기본)/K별 선 그래프를 전환할 수 있고, 전체 선택/선택 해제를 지원합니다.
3D 지도는 단일 결과 선택으로 유지하며 수치 그래프의 복수 선택과 독립적입니다.
실행 파라미터는 GREEDY 선택 설정, MILP 풀이 설정, MAPPO 학습 설정으로 구분합니다.
실패한 작업은 오류로 남기며 그래프에 0점으로 넣지 않습니다. 나머지 조합은 계속 실행합니다.
`남은 조합 중단`은 현재 실행을 끝낸 뒤 다음 작업부터 생략합니다.
다운로드 JSON은 `placement_ui_comparison_v1`이며 공통 설정과 전체 조합별 결과/실패/미실행 상태를 담습니다.
페이지를 닫으면 이후 조합은 실행되지 않습니다. 현재 서버에서 계산 중인 작업은 계속됩니다.
가중치만 변경하면 raycasting만 재사용하며 MAPPO 신경망과 optimizer는 실행마다 새로 초기화합니다.

| 방법 | UI 실행량 설정 | 의미 |
|---|---|---|
| GREEDY | 최대 선택 단계 0~100, 기본 100 | 실제 선택 상한은 min(K, 단계 수). 추가 이득이 없으면 조기 종료. 학습 아님 |
| MILP | 제한시간 1~3600초, 기본 30초; relative gap 0~1, 기본 0.05 | solver 종료 조건. 학습 아님. 전처리 시간까지 보장하는 wall-clock 제한 아님 |
| MAPPO | 32~20000 joint steps, 기본 1000; seed | 매번 처음부터 실제 PPO 학습. K=1~32, 후보 1~5000 |

UI MAPPO 구현은 `src/methods/candidate_mappo.py`이며 기존 `SharedMAPPO` 업데이트 코드를 재사용합니다.
기존 도로용 11-action MAPPO v3와는 **다른 후보 선택형 실험 모델**입니다.
고정 장면에서 K개 agent가 각각 후보 ID 또는 미설치를 동시에 선택하는 1-step cooperative bandit입니다.
중복 ID는 한 대로 합치며 team reward는 공통 가중 목적값을 양의 상수로 정규화합니다.
각 배치는 terminal이므로 다음 배치로 GAE를 이어붙이지 않습니다. actor 입력은 agent ID,
critic 입력은 전체 agent ID이며 지도나 가중치를 입력받아 다른 환경으로 전이하는 모델은 아닙니다.
공유 MLP actor, agent별 value head, rollout 64, PPO epochs 4를 사용합니다.
가중치가 달라지면 실행 버튼으로 새 학습을 수행해야 합니다. 기존 checkpoint는 읽지 않습니다.
최고 목적값을 가진 탐색 배치(빈 배치 baseline 포함)를 표시하며 최종 deterministic 정책 평가는 따로 기록합니다.
JSON metadata에 step/seed/PPO update 수, 학습 기록, 최종 정책 결과와 실행시간을 포함합니다.
이 기능은 초기 통합이며 수렴이나 기존 v3와 동등한 성능을 검증한 것은 아닙니다.
checkpoint 저장/이어학습/중단 기능은 아직 없으며 브라우저를 닫아도 시작한 계산은 서버에서 계속됩니다.

MILP는 1 thread이며 Gurobi 라이선스가 필요합니다.
새 공통 solver API는 기존 frontier runner와 달리 자동 CBC fallback을 하지 않습니다.
CLI에서 CBC를 사용하려면 `solver.solver: pulp`를 설정합니다.
UI는 단일 사용자용이며 서버 메모리에 현재 장면 하나를 공유합니다. 여러 탭/사용자의 독립 세션은 미구현입니다.

### 비교 UI 사용 순서

1. 환경을 불러옵니다. 환경을 불러오는 것만으로 학습/최적화를 시작하지 않습니다.
2. 왼쪽 **비교 알고리즘**에서 실행할 방법을 체크합니다.
3. 각 제목 아래 실행량을 설정합니다: **GREEDY · 선택 설정**, **MILP · 풀이 설정**, **MAPPO · 학습 설정**.
4. **센서 예산 → K 범위**, 시작 3, 종료 5를 지정하면 K=3,4,5를 평가합니다. 하나만 평가하려면 단일 K를 선택합니다.
5. 공통 목적 가중치와 중요 영역을 설정하고 **선택 조합 실행**을 누릅니다. 세 방법이면 총 9개 작업입니다.
6. 완료 후 오른쪽 **결과 그래프 선택** 또는 표의 **그래프** 체크박스로 표시할 결과를 고릅니다.
7. 예를 들어 **선택 해제** 후 표에서 `GREEDY K=3`과 `MAPPO K=5`만 체크하면 두 결과를 동시에 비교합니다.
8. **그래프 형식**에서 묶음 막대/선 그래프, **비교 지표**에서 coverage/중복/미관측/목적값을 선택합니다.
9. 표의 **지도** 선택 버튼은 3D에 표시할 배치 하나를 선택합니다. 그래프 복수 선택을 해제하지 않습니다.
10. **결과 JSON 다운로드**로 전체 실행 결과를 보관합니다. 표시에서 제외한 결과도 다운로드에는 포함됩니다.

왼쪽 실행 설정 변경은 기존 결과를 무효화합니다. 오른쪽 결과 선택은 계산 없이 표시만 변경합니다.
알고리즘/K 묶음 체크박스의 일부 선택 표시는 해당 묶음 중 일부 결과만 표시 중이라는 뜻입니다.
새로고침하면 브라우저의 결과가 사라지므로 필요한 결과는 먼저 다운로드해야 합니다.

### K와 MILP 결과 해석

현재 제약은 `설치 수 <= K`입니다. **정확히 K대 설치하는 모드는 아직 없습니다.**
K=5 결과에 4대가 있어도 오류가 아니며, 4대가 5대보다 낫다는 증명도 아닙니다.
GREEDY는 추가 목적값 이득이 없으면 멈추고, MILP는 비용/중복 penalty와 종료 조건에 따라
K 미만 해를 반환할 수 있습니다. MAPPO는 미설치 또는 중복 후보 제안으로 설치 수가 줄 수 있습니다.
coverage만 최대화하면 기존 배치에 센서를 추가해도 coverage는 감소하지 않지만,
중복/설치 비용 penalty를 포함한 종합 목적값은 감소할 수 있습니다.

MILP의 30초는 의무 실행시간이 아니라 solver 제한시간입니다.
기본 relative MIP gap 0.05는 **가중 목적값의 허용 gap 5%**이며 coverage 5%p 오차가 아닙니다.
예를 들어 양의 목적값에서 현재 해가 100, 최적값 상한이 104이면 relative gap은 4%입니다.
gap 조건을 만족하면 30초 전에 종료할 수 있습니다. gap=0을 지정해도 시간 제한이 먼저 오면 최적성을 보장하지 않습니다.
비교표의 **완료**는 요청 성공을 뜻하며 최적성 증명이 아닙니다.
각 MILP 결과의 `metadata.solver_status`, `mip_gap`, `objective`, `best_bound`, `solve_time_sec`를 함께 확인합니다.
최적성은 주어진 이산 후보 집합과 목적함수 기준이며, 모든 연속 위치/자세에 대한 최적성을 뜻하지 않습니다.

### 입력 형식과 후보 조건

| 경로 | 입력 및 조건 |
|---|---|
| 기존 Roadside CLI | experiment.yaml, parametric scenario, 기존 top-10 후보 |
| 새 Roadside CLI | parametric scenario, objective 및 optimizer 설정 |
| 도로 UI | XY=10m, Z=3m, yaw=55도, pitch=20도, top-M 해제된 별도 demo |
| Robot CLI | box 또는 platform_mesh, 승인된 mount_poses, task target |
| 사용자 mesh UI | mesh 정점·삼각형·승인된 후보 pose·target을 포함한 JSON |

UI 도로 예제와 공식 실험은 후보 조건이 다릅니다. UI 수치를 기존 논문 실험과 혼합하지 않습니다.
사용자 파일 예제는 [configs/mesh_example.json](configs/mesh_example.json)입니다.

```text
vertices:   [V,3] mesh 정점, metres
triangles:  [F,3] 0-based 정점 index
poses:      [M,5] x,y,z,yaw,pitch (m, degree)
targets:    [N,3] 관측 target
importance: [N] 비음수 중요도 (생략 시 1)
costs:      [M] 비음수 설치 비용 (생략 시 0)
sensor:     name, h_fov_deg, v_fov_deg, max_range_m
```

업로드는 최대 12MiB, 후보 2,000개, target 20,000개, 후보×target 2천만 이내입니다.
mount 후보는 사용자가 승인한 센서 원점이며 차체 외부에 있어야 합니다.
장착면 자동 추출, 하우징 충돌 검사, 탑승자 동적 가림은 미구현입니다.
OSM이나 이미지 지도를 직접 업로드하는 기능도 아직 없습니다.
Robot CLI의 STL/OBJ mesh 로드와 UI의 JSON 업로드는 서로 다른 입력 경로입니다.

### 공통 목적함수와 알고리즘

```text
d_j = 선택된 센서 중 target j를 보는 센서 수
c_j = 1[d_j >= 1]
J = w_cov * sum(c_j)
  + w_importance * sum(importance_j * c_j)
  - w_overlap * sum(max(d_j - 1, 0))
  - w_count * selected_sensor_count
  - w_cost * sum(selected_installation_cost)
coverage_pct = 100 * sum(c_j) / target_count
```

가중치 단위는 target 수 기반이며 자동 정규화하지 않습니다. 중요도 배수는 기본 중요도에 곱합니다.
J와 비가중 coverage를 별도로 표시합니다. w_cov=1, 나머지=0이면 기존 union 목표입니다.
모든 가중치는 비음수입니다. 센서 수 penalty가 크면 K보다 적은 센서 또는 0개를 선택할 수 있습니다.
GREEDY는 J의 한 단계 증가량을 최대화하는 heuristic입니다.
MILP는 `sum(max(d-1,0))=sum(d)-sum(c)` 관계를 이용해 같은 목적값을 선형식으로 계산합니다.
새 weighted solve는 기존의 K=1/full-coverage 단축 처리를 적용하지 않습니다.
가중 Roadside 후보 생성은 top-M과 coverage-only dedup을 해제하므로 계산량이 커질 수 있습니다.
가중치에 맞게 후보를 다시 생성해야 하는 경우와 같은 후보에서 재평가하는 경우를 구분해야 합니다.

`robustness_weight != 0`은 아직 명시적 오류입니다. 연속된 사각지대 크기,
localization 오차, 회피 성공률을 이 목적값으로 검증했다고 해석하면 안 됩니다.
공통 `SensorPose`에는 roll 필드가 있지만 legacy pose 변환은 roll=0만 허용하고 광학 모델은 5-DoF입니다.

### 실행 및 결과 호환성

```bash
.venv/bin/python main.py robot --config configs/robot.yaml --dry-run
.venv/bin/python main.py robot --config configs/robot.yaml --result-json results/robot_demo/result.json
.venv/bin/python main.py roadside --config configs/experiment.yaml --dry-run
```

새 Roadside CLI 기본값은 scenario 0/K=1/GREEDY입니다. 기존 scenario_ids/k_values 전체를 자동 실행하지 않습니다.
`roadside.scenario_id`, `optimizer.budget_k`, `optimizer.method`로 변경합니다.
Robot의 중요도/비용은 `robot.target_weights`, `robot.installation_costs`;
Roadside는 `target_importance`, `installation_costs` 배열입니다.
새 CLI의 `placement_v1`과 UI 비교 다운로드의 `placement_ui_comparison_v1`은 기존 연구 result schema와 다릅니다.
UI 비교 파일의 `runs[].result`에 각 단일 결과(`placement_ui_v1`)가 들어갑니다.
`settings`는 공통 가중치/영역/실행량이며 `runs[]`는 알고리즘, K, 완료/실패/미실행 상태와 결과를 포함합니다.
성공 결과에는 pose, target별 관측 밀도, coverage/중복/미관측 지표와 solver/학습 metadata가 있습니다.
기존 compare/view_result에 그대로 넣을 수 없습니다. UI JSON 재업로드도 아직 지원하지 않습니다.
재현하려면 원본 mesh JSON, 사용한 설정·코드와 결과 JSON을 함께 보관해야 합니다.

### 확인된 검증과 다음 단계

- UI 실행량/새 MAPPO 학습 검증: `.venv/bin/python tests/test_placement_ui.py` (4 tests).
- 비교 UI 브라우저 검증: `.venv/bin/python tests/test_placement_comparison.py` (2 tests, Playwright Chromium/Gurobi 라이선스 필요).
- 브라우저 테스트는 Box 예제의 3개 알고리즘 × K=3~5 실행, 개별/묶음 결과 선택, 그래프 전환, JSON 다운로드, 실패값 누락 처리를 검사합니다.
- 이 짧은 실행은 UI 통합 검증이며 MAPPO 수렴 또는 알고리즘 성능 우위의 증거가 아닙니다.
- Roadside s0/K=1: target 7,200개, 후보 7,985개, pose `[30.25,44.25,3,85,-20]`, 관측 4,205개 동일.
- 전체 후보 mask hash 동일, weighted MILP 전수조사 비교 및 Robot 자기 가림 테스트 통과.
- UI 중요 영역 변경으로 선택 후보 변경, visibility 재사용, desktop/mobile 브라우저 확인.

```bash
.venv/bin/python -m compileall -q main.py placement_ui.py src tests
.venv/bin/python tests/test_pipeline.py
.venv/bin/python tests/test_common_placement.py --roadside-parity
.venv/bin/python tests/test_placement_ui.py
.venv/bin/python tests/test_placement_comparison.py
```

후속 작업은 지도·가중치 조건부 RL, custom map/OSM 변환, 실제 플랫폼 장착 제약,
6-DoF 센서 모델, localization/회피 성능 검증입니다. 기존 MAPPO v3는 scenario/K별 학습이며 UI 정책이 아닙니다.

## 최신 MAPPO v3 결과

`results/mappo_v3_official_20260928_130911/manifest.json` 기준으로 2026-09-28
13:10:29~18:25:28(KST)에 180개 run을 완료했습니다. 시나리오 0,1,2,3,6,9,
K=1..6, 20,000 joint steps, 조건별 5 trials입니다. failed_scenarios는 빈 배열입니다.
실행 당시 명령과 GPU 분배는 해당 폴더의 `run.sh`, 설정은 `config.yaml`,
소스 정보는 `source_sha256.txt`, 의존성은 `packages.txt`에서 확인합니다.
기존 폴더에 launcher를 다시 실행하면 provenance 파일을 덮어쓸 수 있으므로 새 결과 폴더를 사용합니다.

동일 범위 재실행 형식(새 tag/root, 순차 실행):

```bash
.venv/bin/python main.py mappo \
  --config results/mappo_v3_official_20260928_130911/config.yaml \
  --scenario-ids 0,1,2,3,6,9 --k-values 1,2,3,4,5,6 \
  --episodes 20000 --trials 5 --tag v3_repeat \
  --output-root results/v3_repeat
```

| 자료 | 경로 |
|---|---|
| 결과 해석 | [V3_RESULTS_SUMMARY.md](results/mappo_v3_official_20260928_130911/V3_RESULTS_SUMMARY.md) |
| trial 원자료 | [mappo_v3_trials.csv](results/mappo_v3_official_20260928_130911/tables/mappo_v3_trials.csv) |
| scenario/K 평균·표본 SD | [mappo_v3_summary.csv](results/mappo_v3_official_20260928_130911/tables/mappo_v3_summary.csv) |
| 기존 baseline 비교 | [s0~2 비교 CSV](results/mappo_v3_official_20260928_130911/tables/mappo_v3_vs_existing_baselines_s0_2.csv) |
| 비교 그림 | `results/mappo_v3_official_20260928_130911/figures/summary/` |

36개 조건 평균 deployment coverage는 86.836%, best deterministic evaluation 평균은 83.109%입니다.
deterministic 값은 평가 이력 중 최고값으로, 최종 checkpoint 한 번의 평가와 같지 않습니다.
대표 deployment는 학습 중 최고 배치와 평가 배치 및 yaw 후처리 후보 중 최고값입니다.
180개 대표 pose 재평가에서 기록 coverage 불일치 0건이라는 검증 기록이 있습니다.
기존 baseline과의 비교 범위는 s0~2/K1~6이며 평균 차이는 GREEDY 대비 +0.450%p,
MILP 대비 -1.417%p입니다. 후보 공간과 실험 버전이 달라 우월성 입증으로 해석하지 않습니다.
CSV/summary 그림은 당시 집계로 생성됐으며 그 전용 집계 스크립트는 아직 정식 CLI에 통합되지 않았습니다.
아래 12~13절의 표와 발표자료는 과거 v2 세미나 기록입니다.

### 먼저 볼 자료

- [현재 세미나 보고서](results/seminar_20260919/SEMINAR_REPORT.md)
- [발표 PPT 18장](results/seminar_20260919/SEMINAR.pptx) / [PDF](results/seminar_20260919/SEMINAR.pdf)
- [회전 가능한 3D 결과 목록](results/seminar_20260919/index.html)
- [전체 K별 결과 CSV](results/seminar_20260919/tables/coverage_summary.csv)
- [실제 실행 설정](results/seminar_20260919/config.yaml) / [manifest](results/seminar_20260919/manifest.json)

### 읽는 순서

| 절 | 내용 |
|---|---|
| 1~3 | 연구 목적, 파일 역할, 전체 실행 흐름 |
| 4~7 | 설정, 시나리오, 설치 영역, LiDAR, raycasting, coverage 수식 |
| 8~10 | MILP / GREEDY / MAPPO 구성과 학습 |
| 11~12 | 결과 schema, 통계 처리, 현재 실제 결과 |
| 13~14 | 3D/세미나 자료와 실행 명령 |
| 15~16 | 검증 범위, 재현성, 남은 연구 한계 |
| 17 | 의존성과 이용 조건 |

알고리즘 핵심 파일은 분리되어 있습니다. v3 전체 실행은 완료됐으며 기존 v2 세미나 기록도 별도로 보존되어 있습니다.

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
│   ├── experiment.yaml
│   ├── robot.yaml               # 플랫폼 예제
│   └── mesh_example.json        # UI 사용자 mesh 입력 예제
├── data/
│   ├── scenarios/
│   │   └── yeongjong_final.json
│   ├── cache/                   # raycasting visibility cache, git 제외
│   ├── gurobi_nodefiles/        # Gurobi 임시 node file, git 제외
│   └── osm/                     # 현재 계산에는 사용하지 않음
├── src/
│   ├── core/                    # candidate, models, objective, evaluation, visibility, mesh_problem
│   ├── roadside/                # candidate_generator, problem_builder, visibility_adapter
│   ├── robot/                   # 플랫폼 pipeline
│   ├── application.py           # 새 roadside/robot CLI adapter
│   ├── environment.py           # 설정, 지도, LiDAR, raycasting, coverage
│   ├── outputs.py               # JSON/NPY 저장, 로드, 비교 CSV
│   ├── visualize.py             # top-view, heatmap, 3D snapshot, 학습 그래프
│   └── methods/
│       ├── milp.py              # 공통 MILP, 기존 budget frontier와 호환 import
│       ├── greedy.py            # greedy baseline
│       └── mappo.py             # native MAPPO port와 학습 환경
├── tests/
│   ├── test_pipeline.py         # 기존 파이프라인 무결성
│   ├── test_common_placement.py # 공통 objective/회귀/가림
│   ├── test_placement_ui.py     # UI API/가중치/입력 검증
│   └── test_placement_comparison.py # 복수 비교/선택/다운로드 브라우저 검증
├── placement_ui.py              # Flask 로컬 UI 서버
├── ui/index.html                # 비교 UI 구조
├── static/workbench.js          # 알고리즘/K 순차 실행, 비교 그래프, 3D 전환
├── REFACTOR_PROGRESS.md         # 리팩터링 구현 범위
├── seminar.py                   # 세미나 실험/재검산/비교/3D/PPT runner
├── main.py                      # 통합 CLI
├── test.py                      # 안전 실행 runner, --execute 없으면 계획만 출력
├── view_result.py               # Matplotlib/Open3D 대화형 결과 viewer
├── README.md
└── LICENSE
```

### 2.1 Quick Start

Linux:

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

경량 검증:

```bash
python -m py_compile main.py test.py seminar.py view_result.py src/environment.py src/outputs.py src/visualize.py src/methods/milp.py src/methods/greedy.py src/methods/mappo.py tests/test_pipeline.py
python tests/test_pipeline.py
```

전체 실행은 계산량이 큽니다. Gurobi backend를 사용하려면 유효한 라이선스가 필요하며, 사용할 수 없을 경우 작은 문제에 대해 PuLP/CBC fallback을 시도합니다.

최신 MAPPO 결과는 `results/mappo_v3_official_20260928_130911/`이며 `seminar_20260919/`는 과거 v2 기록입니다. 결과가 없는 clone에서는 해당 실행 명령과 설정 snapshot이 필요합니다.

### 2.2 파일별 역할과 경계

| 파일 | 책임 |
|---|---|
| `src/environment.py` | config/scenario 로드, MapGeometry, LiDAR profile, target, visibility cache, evaluator |
| `src/core/` | 공통 데이터, 목적함수, 평가, 임의 mesh visibility |
| `src/roadside/` | 기존 순서/pruning/dedup을 보존한 후보 생성과 공통 문제 변환 |
| `src/robot/pipeline.py` | 플랫폼 mesh, 승인된 mount pose와 task target 연결 |
| `src/methods/milp.py` | 공통 sparse MILP, 기존 budget frontier |
| `src/methods/greedy.py` | 공통 marginal objective gain 선택과 기존 runner |
| `src/methods/mappo.py` | 학습 state/action/reward, actor-critic, rollout/GAE/PPO, best pose |
| `src/outputs.py` | JSON/NPY schema·저장·검색, 일반 compare CSV |
| `src/visualize.py` | 저장 결과의 topview/heatmap/3D PNG/학습 그래프 |
| `main.py` | 방법별 실행과 compare/visualize CLI |
| `test.py` | `--execute` 없으면 계획만 출력하는 안전 runner |
| `seminar.py` | 이번 고정 시나리오 실험 및 CSV/HTML/PPT 생성 |
| `view_result.py` | Matplotlib/Open3D GUI viewer |
| `tests/test_pipeline.py` | 경량 단위·회귀 테스트 |
| `requirements.txt` | Python 의존성. 완전한 버전 lock은 아님 |
| `src/__init__.py` 등 | Python package 구성 |

`test.py`는 단위 테스트 파일이 아닙니다. 과거 shell script가 남아 있어도 현재 실행 경로는 이 문서의 명령을 기준으로 합니다.
`data/cache/`는 계산 재사용용이고 `data/gurobi_nodefiles/`는 solver 임시 파일입니다. 코드·시나리오와 구분하세요.

## 3. 전체 파이프라인

```text
CLI 인자 + YAML
       │
       ▼
scenario ID → topology / road_width / num_lanes
       │
       ▼
MapGeometry: 도로·인도·건물·중앙부 mesh
       ├─ 설치 가능 XY grid + Z levels
       └─ 도로 target points
                    │
                    ▼
visibility cache 검증 → 없으면 Open3D raycasting → packed mask 저장
                    │
         ┌──────────┴─────────────────┐
         ▼                            ▼
MILP/GREEDY 후보 builder          MAPPO 초기 spawn
XY/Z/yaw/pitch 조합               state/action mask/reward
FOV → top-10 → mask dedup         shared actor / global critic
         │                            │
         ├─ GREEDY marginal gain      └─ PPO → 평가 → best pose → yaw tuning
         └─ MILP budget frontier                    │
         └────────────────┬─────────────────────────┘
                          ▼
              NPY pose + JSON metadata + MAPPO NPZ log
                          │
                          ▼
          최종 배치 재검산 → CSV → 비교/학습/시간 그래프
                          │
                          ▼
        topview / heatmap / 3D PNG / 대화형 HTML / 세미나 PPT
```

### 3.1 실행 단위

MILP/GREEDY는 `scenario × K`, MAPPO는 `scenario × K × trial`입니다.
MILP/GREEDY는 scenario별 candidate 집합을 여러 K에 재사용합니다.
MAPPO는 scenario/K/trial마다 새 정책을 만들며, K가 바뀌면 critic 차원도 바뀝니다.

기본 YAML 전체 실행은 MILP 36개, GREEDY 36개, MAPPO 180개입니다.
**이번 세미나는 범위를 줄여 MILP 9개, GREEDY 9개, MAPPO 27개를 실행했습니다.**

### 3.2 공통 환경과 학습용 환경의 차이

`LegacyMAPPOEnv`라는 이름은 이전 코드에서 가져온 구조를 뜻합니다.
별도 legacy main을 호출하는 black-box가 아니라 `MapGeometry`, 공통 cache/profile에 기반한 학습 상태·보상 관리 클래스입니다.

다만 MAPPO에는 GPU용 Torch FOV/reward 계산 경로가 따로 존재합니다. 공통 NumPy evaluator와 수정 사항이 자동 동기화되는 것은 아니므로, 물리 모델 수정 시 두 경로를 검증해야 합니다.
최종 저장은 `evaluate_coverage()`로 재평가합니다.
MILP/GREEDY는 공통 `coverage_mask_for_pose()`에서 만든 mask union을 사용합니다.

### 3.3 seminar.py 실행 순서

1. 기존 output root가 있으면 실패시켜 결과 혼합을 막습니다.
2. 기본 YAML을 읽어 scenario=[0,1,2], K=[1,2,3], 지정 steps/trials와 output root를 snapshot `config.yaml`에 씁니다. 원본 YAML은 바꾸지 않습니다.
3. manifest/package 목록/source diff를 기록합니다.
4. scenario별 worker를 병렬 실행합니다. 각 worker는 GREEDY → MILP → MAPPO 순서로 실행합니다.
5. `scenario_N.log`, `timing_sN.json`을 기록합니다.
6. worker 실패 시 실패 상태를 기록하고 보고서를 생성하지 않습니다.
7. 모두 성공하면 report가 기대 조합과 중복, 최종 coverage를 검사하고 CSV/그림/HTML/PPT를 생성합니다.

현재 runner는 GPU 0/1/2를 worker에 지정합니다. GPU 구성이 다른 머신에서는 일반 CLI로 순차 실행하세요.
scenario/K는 seminar.py 상수이므로 임의 조합은 `main.py`로 실행합니다.
`seminar.py report`는 학습을 재실행하지 않지만 기존 표/그림/PPT를 다시 생성합니다.

### 3.4 설정 우선순위

CLI의 scenario/K/profile/episodes/trials는 해당 run 범위를 덮어씁니다.
특히 `main.py`의 `--output-root` 기본값은 `results/active`여서 YAML output을 덮어쓸 수 있습니다. 실험마다 명시하세요.
viewer/report에는 현재 기본 config가 아니라 **그 결과를 생성한 snapshot config**를 전달합니다.

## 4. 실험 설정

기본 설정 파일은 `configs/experiment.yaml`이며 이번 실제 설정은 `results/seminar_20260919/config.yaml`입니다.

| 항목 | 기본 YAML | 세미나 실행 |
|---|---|---|
| scenario | 0,1,2,3,6,9 | 0,1,2 |
| K | 1~6 | 1~3 |
| MAPPO steps | 20,000 | 5,000 |
| trials | 5 | 3 |
| output | results/active | results/seminar_20260919 |

아래 parameter 설명은 기본 YAML 기준이며 세미나에서는 위 범위만 변경했습니다.

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
  ppo_lr: 0.0003
  ppo_gamma: 0.999
  ppo_eps_clip: 0.15
  ppo_k_epochs: 5
  ppo_update_timestep: 256
  ppo_entropy_coef: 0.003
  ppo_gae_lambda: 0.95
  ppo_value_clip: 0.2
  ppo_minibatch_size: 128
  ppo_reward_scale: 0.01
  ppo_reward_clip: 10.0
  eval_interval: 100
  eval_steps: 50
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
| `eval_interval` | 100 | fixed-start deterministic 평가 간격 |
| `eval_steps` | 50 | 같은 초기 pose에서 수행하는 deterministic 평가 horizon |
| `plateau_patience` | 200 | plateau 판단 step 수 |
| `plateau_entropy_scale` | 0.3 | plateau 시 entropy scale |
| `plateau_uncovered_scale` | 0.3 | plateau 시 uncovered reward scale |
| `print_interval` | 20 | 로그 출력 및 중간 NPZ 저장 간격 |

`ppo_update_timestep=256`은 joint environment step 256개를 뜻합니다. 각 rollout tensor는 `[256, K, ...]`이고 actor 학습 batch에는 최대 `256 x K`개 transition이 들어갑니다.

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

### 5.1 기본 설정의 시나리오

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

기존 Roadside 실험은 동일 target weight를 사용합니다. 새 공통 엔진/UI는 사용자 중요도를 지원하지만 교통량·차로·객체 속성을 자동으로 중요도에 변환하는 기능은 없습니다.

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

### 6.5 Cache 검증과 이산 pose의 범위

기존 cache는 key, config hash, visibility model, table을 검사하여 재사용합니다. metadata가 맞지 않으면 오류를 내고, 파일이 없으면 생성합니다. `bake_visibility_map(..., force=True)`는 강제 재생성 API입니다.

hash에는 scenario/profile/candidate/target 및 visibility 관련 일부 MAPPO 설정이 들어갑니다. 모든 source 변경과 `MapGeometry` 상수를 자동으로 포함하지는 않으므로 geometry 수정 시 model version과 cache invalidation을 관리해야 합니다.
현재 cache write는 atomic하지 않습니다. 같은 cache를 여러 worker가 동시에 처음 생성하는 실행은 피하고, 신뢰할 수 없는 pickle을 로드하지 마세요.

`coverage_mask_for_pose()`는 XY를 grid cell로, Z를 가장 가까운 level로 매핑합니다. 임의 연속 좌표의 정확한 raycast를 매번 수행하는 함수가 아닙니다. 허용 이산 pose인지 별도로 검증해야 합니다. `is_installable()`만 호출하면 높이 grid까지 검사한 것은 아닙니다.

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

`road_detection`을 2 이상으로 바꾸면 현재 union 기반 MILP/GREEDY formulation과 달라집니다. YAML 값만 바꾸고 동일 목적함수 비교라고 간주하면 안 됩니다.

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

`max_orientations_per_position: 0`은 pruning을 끄지만 계산량과 메모리가 커집니다.
전역 mask dedup은 서로 다른 위치라도 coverage가 같으면 하나를 남길 수 있습니다. 이후 pole별 제약/비용을 추가하면 이 규칙도 검토해야 합니다.

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

현재 formulation에는 같은 pole당 한 대 같은 별도 occupancy 제약이 없습니다. 센서 수 제약과 실제 설치 시설 제약은 구분해야 합니다.

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

GREEDY와 MILP의 기존 runner는 `src.roadside.candidate_generator.build_oriented_candidates()`를 공유합니다. `src.methods.milp`의 기존 import 경로도 호환됩니다. 아래는 기존 비가중 선택 규칙입니다.

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
→ Linear(K agent values)
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

buffer 저장 전 reward에 `ppo_reward_scale=0.01`을 곱하고 `[-10, 10]`으로 clipping합니다. 이 scale은 advantage 정규화 전 critic target의 크기를 낮추기 위한 설정입니다.

### 10.7 PPO update와 수정된 rollout

현재 PPO는 다음 요소를 포함합니다.

- old policy sampling
- clipped surrogate objective
- centralized value loss
- clipped value update
- GAE
- entropy bonus와 decay
- actor/critic 별도 Adam optimizer
- actor/critic 개별 gradient norm clipping 1.0
- minibatch update

현재 source는 joint transition을 `[time, agent, ...]` 형태로 저장합니다. old/new log probability는 flatten 후 모두 `[B]`인지 검사하며, GAE는 agent별 열을 유지한 채 시간축으로만 계산합니다. rollout 마지막은 다음 global state의 centralized critic 값으로 bootstrap합니다.

critic은 global state에서 K개 agent value를 출력합니다. PPO minibatch에서는 각 `(time, agent)` sample에 해당하는 value head를 선택합니다. `ppo_random_action_prob`도 별도 off-policy 분기가 아니라 old/new policy 모두에서 동일한 categorical mixture distribution으로 계산합니다.

ratio/GAE 수정 결과는 `mappo_native_port_rollout_v2`로 기록되었습니다. 기존 세미나 MAPPO는 v2이며 평가·저장 provenance와 actor/critic optimizer를 수정한 v3의 전체 180개 실행도 완료되었습니다. 결과 폴더를 구분해야 합니다.

```text
delta[t,i] = r[t,i] + gamma × (1-done[t]) × V_next[i] - V[t,i]
A[t,i] = delta[t,i] + gamma × lambda × (1-done[t]) × A[t+1,i]
return[t,i] = A[t,i] + V[t,i]

ratio = exp(log pi_new(a|s) - log pi_old(a|s))
L_policy = -mean(min(ratio × A, clip(ratio,1-eps,1+eps) × A))
L_actor  = L_policy - entropy_coef × entropy
L_critic = clipped_value_loss
```

수정한 문제는 ratio의 [B,B] broadcasting, agent 간 trajectory 혼합, continuing-task update 끝의 잘못된 zero bootstrap입니다.
GAE 후에만 time/agent 축을 minibatch로 펼칩니다. critic의 해당 agent value head를 선택합니다. actor와 critic은 파라미터를 공유하지 않으므로 별도 optimizer와 별도 gradient clipping을 사용합니다.
256은 agent sample 수가 아니라 joint timestep 수입니다. 5,000 steps에서는 약 5,000K개 agent transition과 별도의 deterministic 평가 이동이 발생합니다.

### 10.8 Best pose 저장

학습 중 stochastic trajectory의 최고 pose와 deterministic policy 평가 pose를 서로 다른 NPY에 저장합니다. 정책 평가는 100 step마다 항상 동일한 초기 pose로 돌아가 50 action을 수행하며, 평가 중 uncovered hint를 자체 갱신하고 종료 후 학습 agent/environment 상태를 복원합니다. 마지막 residual PPO update 뒤에도 최종 평가를 다시 수행합니다.

학습 종료 후 stochastic best와 policy-evaluation best 각각에 coordinate-wise yaw tuning을 적용하고 별도 NPY로 보존합니다. 대표 `coverage_pct`는 네 후보 중 공통 coverage가 가장 높은 deployment pose이며, `extra.reported_solution_source`로 출처를 기록합니다. 따라서 대표 수치는 정책 return이 아니라 RL 탐색과 선택적 yaw 후처리를 포함한 배치 최적화 결과입니다.

마지막으로 대표 pose를 공통 `evaluate_coverage()`로 다시 평가하고 내부 evaluator와 불일치하면 실패합니다.

### 10.9 실제 학습 순서와 재개 한계

1. Python/NumPy/Torch seed 설정. 현재 seed는 `seed_base + trial`입니다.
2. `LegacyMAPPOParams`, 공통 환경 기반 `LegacyMAPPOEnv`, 초기 spawn 생성.
3. `SharedMAPPO`와 K개 `PPOAgent` 생성.
4. joint state 수집 → agent 행동 → reward 계산 → next global state 수집.
5. joint transition 저장 → 256 steps마다 PPO update → 마지막 남은 rollout update.
6. 고정 초기 pose에서 deterministic policy 평가 → stochastic/eval raw pose를 분리 저장.
7. 두 raw pose에 yaw tuning 적용 → 네 후보 중 대표 deployment pose 선택.
8. 최종 NPY를 공통 evaluator로 재검산하고 JSON/NPZ/PT 저장.

`*_policy.pt`에는 최종 old policy 가중치와 구조 metadata를 저장합니다. optimizer를 포함한 완전한 resume checkpoint는 아닙니다.
같은 run의 산출물이 하나라도 존재하면 학습을 거부합니다. 재실행할 때는 반드시 새 tag/root를 사용해야 하며, 과거 NPY를 새 run의 best로 자동 혼합하지 않습니다.
현재 actor도 미커버 centroid 같은 global hint를 입력받습니다. 순수 local-only 관측 정책으로 설명하지 않습니다.

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
extra.solver_objective
extra.solver_best_bound
extra.solve_time_sec
extra.candidate_count
```

### 11.4 MAPPO 추가 필드

```text
train_log_npz
extra.trainer
extra.rollout_layout
extra.training_steps
extra.training_config_hash
extra.ppo_updates
extra.ppo_lr
extra.ppo_update_timestep
extra.ppo_reward_scale
extra.ppo_reward_clip
extra.legacy_action_dim
extra.result_semantics
extra.reported_solution_source
extra.reported_solution_postprocessed
extra.policy_evaluation_start
extra.policy_evaluation_steps
extra.best_train_raw_coverage_pct
extra.best_eval_raw_coverage_pct
extra.best_train_yaw_tuned_coverage_pct
extra.best_eval_yaw_tuned_coverage_pct
extra.best_internal_saved_coverage_pct
extra.best_internal_saved_score
extra.train_raw_positions_npy
extra.eval_raw_positions_npy
extra.train_yaw_tuned_positions_npy
extra.eval_yaw_tuned_positions_npy
extra.policy_checkpoint
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
ppo_update
ppo_environment_step
ppo_rollout_steps
ppo_samples
ppo_policy_loss
ppo_value_loss
ppo_entropy
ppo_approx_kl
ppo_clip_fraction
ppo_actor_grad_norm
ppo_critic_grad_norm
ppo_explained_variance_before_update
ppo_entropy_coef
```

### 11.6 일반 compare와 세미나 통계의 차이

| 항목 | main.py compare | seminar.py report |
|---|---|---|
| 구현 | src/outputs.py | seminar.py |
| CSV | compare_methods.csv | raw_results/coverage_summary/runtime.csv |
| MAPPO SD | 모집단 ddof=0 | 표본 ddof=1 |
| 완전성/중복 검사 | 기대 조합 강제 검증 없음 | 현재 45개 조합 검사 |
| coverage 재검산 | JSON 수치 집계 | NPY를 공통 evaluator로 재평가 |
| scenario/K filter | CLI 인자는 있지만 실제 row filtering 미적용 | 0/1/2, K=1/2/3 고정 |

일반 compare는 한 root에 한 실험만 저장하여 사용합니다. profile/tag/config가 섞이면 잘못 집계될 수 있습니다.
세미나의 공식 집계 파일은 `tables/coverage_summary.csv`입니다.
deterministic method의 SD가 비어 있는 것은 반복 trial이 없다는 의미입니다.

### 11.7 Runtime

- GREEDY/MILP: scenario 내 K=1..3 frontier 전체 wall time, 후보 생성 포함.
- MAPPO: scenario/K/trial 한 run의 초기화, 학습, 평가, yaw 튜닝, 저장 포함.
- scenario worker 병렬 실행 및 기존 visibility cache 재사용 조건입니다.
- runtime plot은 method별 **총 실험 비용**입니다. MILP/GREEDY 3개 budget과 MAPPO 3 budgets × 3 trials는 작업량이 다릅니다.
- cache 최초 생성, training, inference-only 시간을 구분해야 하며 현재 그래프를 동등 예산 speedup으로 해석하면 안 됩니다.

## 12. 과거 v2 세미나 결과

실행일 2026-09-19, 자료 갱신일 2026-09-21.
시나리오 0/1/2, K=1/2/3, MAPPO 5,000 joint steps × 3 trials.
GREEDY 9개, MILP 9개, MAPPO 27개, 총 45개 결과입니다.

### 12.1 Coverage

단위 %, MAPPO는 평균 ± 표본 표준편차입니다.

| scenario | K | GREEDY | MILP frontier | MAPPO |
|---|---:|---:|---:|---:|
| 0 직선 | 1 | 58.403 | 58.403 | 58.384 ± 0.032 |
| 0 직선 | 2 | 98.667 | 98.667 | 98.162 ± 1.209 |
| 0 직선 | 3 | 99.875 | 99.875 | 99.866 ± 0.016 |
| 1 T자 | 1 | 60.092 | 60.092 | 57.747 ± 3.239 |
| 1 T자 | 2 | 79.628 | 79.628 | 75.031 ± 1.891 |
| 1 T자 | 3 | 90.613 | 90.613 | 92.679 ± 2.897 |
| 2 십자 | 1 | 55.086 | 55.086 | 54.847 ± 0.004 |
| 2 십자 | 2 | 71.645 | 71.645 | 69.242 ± 0.462 |
| 2 십자 | 3 | 79.378 | 79.378 | 79.661 ± 2.300 |

출처: [coverage_summary.csv](results/seminar_20260919/tables/coverage_summary.csv).
생성 명령: `.venv/bin/python seminar.py report --root results/seminar_20260919`.

이 과거 300초 제한 실행에서는 K=3 T자/십자의 MAPPO 평균이 저장 MILP frontier보다 높았습니다.
이후 1,200초 실행(`seminar_k11_20260921`)의 MILP는 T자 K=3에서 99.746%, 십자 K=3에서 85.014%로 바뀌었습니다. 과거 표를 현재 우열 결론으로 사용하지 않습니다.
후보 공간 차이, best pose 선택, yaw 후처리, solver 제한 때문에 전역 최적 MILP를 능가했다는 주장도 불가능합니다.

### 12.2 MILP 상태

| 조합 | 상태 | 해석 |
|---|---|---|
| 모든 scenario K=1 | SKIPPED_EXACT_K1 | pruned 후보 내 단일 최대값 |
| 직선 K=2,3 | 2 | 설정 tolerance 내 종료 |
| T자/십자 K=2,3 | 9 | 300초 시간 제한 |

직선 K=2,3의 gap은 약 1.35%, 0.125%로 exact optimum 증명은 아닙니다.
시간 제한 4개는 gap/bound가 `inf`로 저장되어 유한한 최적성 인증이 없습니다.
이번 9개 조합에서 저장 MILP frontier는 GREEDY와 같은 coverage입니다.
bound/status는 raw solve metadata이고 frontier의 출처는 별도 필드를 함께 확인합니다.

## 13. 결과 폴더와 3D 및 세미나 자료

```text
results/seminar_20260919/
  config.yaml / manifest.json / packages.txt / hardware.json
  source.patch / source_snapshot.zip / report_source_hashes.json
  scenario_0.log / scenario_1.log / scenario_2.log
  timing_s0.json / timing_s1.json / timing_s2.json
  greedy/                  9 JSON + 9 NPY
  milp/                    9 JSON + 9 NPY
  mappo/                   27 JSON + 27 NPY + 27 train-log NPZ
  tables/
    raw_results.csv        45개 raw 결과
    coverage_summary.csv   27개 scenario/K/method 집계
    runtime.csv            frontier/run별 시간
  figures/
    coverage_comparison.png / pdf
    learning_curves.png
    runtime.png
    ... 대표 배치 topview / heatmap / 학습 그래프
  3d/                      대표 배치 3D PNG
  interactive_3d/           27개 HTML + plotly.min.js
  representatives.json     대표 trial 및 자료 경로
  validation.json          이번 검증 기록
  index.html               자료/3D 목록
  SEMINAR_REPORT.md
  SEMINAR.pptx / SEMINAR.pdf
```

### 13.1 그래프 의미

- Coverage 비교: scenario별 K에 따른 세 방법의 최종 coverage. MAPPO errorbar는 표본 SD.
- 학습 이력: stochastic rollout의 running-best coverage 평균과 seed별 min/max band. deterministic 평가/yaw 튜닝 제외. 정책 수렴을 의미하지 않음.
- Runtime: method별 총 실험 비용. 서로 작업량이 다름.
- Topview: 위에서 본 배치/관측 영역.
- Heatmap: 관측 분포. 2D 그림과 3D target 집계값을 구분.
- 3D PNG: PPT용 정적 이미지.
- HTML 3D: 환경 mesh, covered/uncovered target, 설치 가능 cell, LiDAR pose와 방향. 회전/확대 가능.

대표 MAPPO trial은 최고가 아니라 **평균 coverage에 가장 가까운 trial**입니다.
HTML은 covered/uncovered target 각각 최대 6,000개를 샘플링하지만 표시 수치는 전체 target 계산입니다.
`plotly.min.js`를 HTML과 같은 폴더에 유지하면 오프라인으로 열 수 있습니다.

### 13.2 Python viewer와 구분

`view_result.py`는 Matplotlib/Open3D GUI viewer입니다.
global coverage는 공통 evaluator를 사용하지만 agent별 visible/unique/overlap과 일부 beam은 거리/FOV만 계산하는 경로가 남아 있습니다.
그 agent별 통계를 논문 수치로 쓰지 마세요. 이번 Plotly HTML covered/uncovered mask는 공통 evaluator 결과입니다.
SSH에서 GUI를 열려면 display 연결이 필요합니다. HTML은 로컬 브라우저로 볼 수 있습니다.

## 14. 실행 명령

모든 명령은 프로젝트 root에서 실행합니다. 새 실험은 새 tag와 새 root를 사용합니다.

### 14.1 실행 전 검증

```bash
.venv/bin/python -m py_compile main.py test.py seminar.py view_result.py src/environment.py src/outputs.py src/visualize.py src/methods/milp.py src/methods/greedy.py src/methods/mappo.py tests/test_pipeline.py
.venv/bin/python tests/test_pipeline.py

.venv/bin/python main.py greedy --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run
.venv/bin/python main.py milp --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --dry-run
.venv/bin/python main.py mappo --config configs/experiment.yaml --scenario-ids 0 --k-values 1 --episodes 10 --trials 1 --tag dryrun --dry-run
```

dry-run은 계획 확인이지 solved result가 아닙니다.
경량 테스트에는 임시 directory의 raycasting과 작은 PPO update가 포함됩니다.

### 14.2 이번 세미나 실험 생성 명령

아래 run은 실제 계산합니다. 현재 root가 존재하므로 재실행하려면 이름을 바꾸세요.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u seminar.py run \
  --root results/seminar_20260919 --steps 5000 --trials 3
```

기존 결과로 자료만 재생성:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python seminar.py report \
  --root results/seminar_20260919

libreoffice --headless --convert-to pdf \
  --outdir results/seminar_20260919 results/seminar_20260919/SEMINAR.pptx
```

PDF 변환에는 LibreOffice가 별도로 필요합니다.
report는 학습/최적화를 하지 않으며 표/그림/PPT만 다시 생성합니다.

### 14.3 일반 CLI로 순차 실행

아래는 새 실행의 예시이며 실제 계산 명령입니다.

```bash
ROOT=results/new_comparison
TAG=new_comparison

.venv/bin/python main.py greedy --config configs/experiment.yaml \
  --scenario-ids 0,1,2 --k-values 1,2,3 --tag "$TAG" --output-root "$ROOT"

.venv/bin/python main.py milp --config configs/experiment.yaml \
  --scenario-ids 0,1,2 --k-values 1,2,3 --tag "$TAG" --output-root "$ROOT"

.venv/bin/python main.py mappo --config configs/experiment.yaml \
  --scenario-ids 0,1,2 --k-values 1,2,3 --episodes 5000 --trials 3 \
  --tag "$TAG" --output-root "$ROOT"

.venv/bin/python main.py compare --config configs/experiment.yaml \
  --scenario-ids 0,1,2 --k-values 1,2,3 --mappo-tag "$TAG" --output-root "$ROOT"
```

`main.py` subcommand는 milp/greedy/mappo/compare/visualize/full입니다.
**full은 scaffold이며 실제 전체 실험을 실행하지 않습니다.**
일반 CLI는 모든 설정/provenance snapshot을 자동 기록하지 않으므로 실제 config와 명령을 별도 보관합니다.
compare의 scenario/K filter는 실제 수집에 적용되지 않으므로 한 root에 한 실험만 저장하세요.

### 14.4 안전 runner

```bash
.venv/bin/python test.py plan --config configs/experiment.yaml \
  --scenario-ids 0 --k-values 1 --episodes 10 --trials 1

# --execute가 있어야 실제 계산
.venv/bin/python test.py greedy --config configs/experiment.yaml \
  --scenario-ids 0 --k-values 1 --tag quickcheck \
  --output-root results/quickcheck --execute
```

### 14.5 저장 결과 시각화

```bash
.venv/bin/python main.py visualize \
  --config results/seminar_20260919/config.yaml \
  --scenario-ids 0,1,2 --k-values 1,2,3 --methods milp,greedy,mappo \
  --mappo-tag seminar_20260919 --results-root results/seminar_20260919 \
  --output-dir results/seminar_20260919/figures \
  --output-3d-dir results/seminar_20260919/3d --save-3d
```

이는 matching trial 전체 export 경로이며 seminar 대표 trial 선택/PPT 생성과 다릅니다.
단일 Python 3D viewer:

```bash
.venv/bin/python view_result.py \
  --config results/seminar_20260919/config.yaml \
  --result-json results/seminar_20260919/greedy/greedy_velodyne64_r30_seminar_20260919_id0_k3.json \
  --viewer matplotlib
```

`--viewer open3d`로 변경할 수 있습니다.
브라우저에서는 `results/seminar_20260919/index.html`을 여세요.

### 14.6 로그 확인

```bash
tail -n 40 results/seminar_20260919/scenario_2.log
tail -f results/seminar_20260919/scenario_2.log
cat results/seminar_20260919/manifest.json
```

`tail -f`를 Ctrl+C로 종료해도 학습은 중단되지 않습니다.
manifest의 complete는 worker와 report 성공을 뜻합니다.
중간 NPZ/NPY만 있고 JSON이 없으면 해당 run의 최종 저장이 끝나지 않았을 수 있습니다.

## 15. 검증과 재현성

### 15.1 이번 실행에서 확인한 범위

- 문법 검사, tests/test_pipeline.py 7 tests, 세 방법 dry-run 통과.
- 테스트: cache/plan, schema, time-agent GAE와 bootstrap, PPO update shape/finite 진단값, fixed-start 평가 상태 복원, 결과 provenance 선택, raycasting smoke.
- 실제 45개 결과의 NPY를 공통 evaluator로 재계산하여 저장 coverage와 covered_count 일치.
- pose 개수/유한값/설치 가능 위치/높이/yaw/pitch 검사 통과.
- MAPPO 27개 NPZ 모두 5,000 steps 및 finite 값 검사 통과.
- HTML 27개 목록, 대표 WebGL canvas와 camera 회전, JavaScript 오류 없음 확인.
- PDF 18장 생성 및 일부 슬라이드 한글/3D 렌더링 확인.

위 27개 MAPPO 결과 검증은 과거 v2 산출물에 대한 것입니다. v3는 초기 진단 run 이후 180개 전체 실행을 완료했고, 대표 pose 재평가에서 coverage 불일치 0건이라는 검증 기록이 있습니다. 이는 학습 수렴·전이 성능·최적성을 증명하지 않습니다.
문서 갱신 과정에서 이 대규모 실험을 재실행하지 않았으며, 위 항목은 이번 실험 정리 시 수행한 검증 기록입니다.

### 15.2 추적 자료

| 파일 | 용도 |
|---|---|
| manifest.json | 명령 인자, 시작/종료, 상태, source hash, git revision |
| config.yaml | 실제 실행 설정 |
| packages.txt | 설치 package 버전 |
| hardware.json | CPU/RAM/GPU/driver |
| source.patch | 실행 시 src/tests diff |
| source_snapshot.zip | 자료 정리 시점 source archive |
| report_source_hashes.json | 자료 생성 source hash |
| validation.json | 최종 배치와 학습 로그 검증 기록 |

자료 생성 코드를 실행 중/후 개선했으므로 최초 manifest source hash와 report source hash는 용도가 다릅니다.
dirty workspace는 git revision만으로 재현할 수 없으므로 diff/snapshot도 필요합니다.
일부 hardware/snapshot/validation 파일은 이번 정리 과정에서 별도 생성했으며 모두 run 명령의 자동 산출물은 아닙니다.
GPU/병렬 계산의 bitwise 재현성은 보장하지 않습니다.
JSON 내부 pose/log 경로가 프로젝트 root 기준일 수 있으므로 자료를 이동한 뒤 경로를 확인하세요.

## 16. 남은 한계와 논문 방향

1. **후보 공간:** top-10 MILP/GREEDY와 전체 orientation MAPPO가 다릅니다. 공통 후보 실험 또는 topM sensitivity가 필요합니다.
2. **MILP 인증:** 시간 제한/gap/bound를 함께 보고해야 합니다. feasible result를 optimum이라고 부르면 안 됩니다.
3. **MAPPO 검증:** v3 전체 180개 실행은 완료됐습니다. policy 평가와 best-seen deployment 수치를 구분하고, shaping/reward-scale/yaw 후처리 ablation과 random/local-search baseline 비교를 추가해야 합니다.
4. **전이:** 현재 scenario/K마다 새로 학습합니다. train/test 맵 분리, zero-shot/fine-tuning 결과는 TBD입니다.
5. **물리 모델:** 64개 실제 channel, point density, 반사율, 기상, calibration error, 차량/보행자 동적 가림은 없습니다.
6. **설치 제약:** 인도 중심 이산 설치선이며 pole별 용량, 전력/통신, 부지/비용을 모델링하지 않습니다.
7. **Cache:** geometry/source version을 완전히 hash하지 않으며 atomic 저장/동시 생성 관리가 더 필요합니다.
8. **통계/결과 혼합:** 일반 compare의 filter와 completeness를 보강해야 합니다. 같은 tag 재사용도 피해야 합니다.
9. **Resume:** policy checkpoint는 평가용 가중치이며 optimizer/rollout 상태를 포함한 학습 재개 기능은 아닙니다.
10. **Viewer:** 일부 agent별 통계는 occlusion 미반영이므로 공통 evaluator로 통일할 필요가 있습니다.

현재 가능한 설명은 “정적 파라메트릭 도로에서 공통 geometric coverage를 이용한 세 배치 방법의 예비 비교”입니다.
실제 교통 인지 성능 보장이나 exact MILP보다 우수하다는 주장을 뒷받침하지 않습니다.

후속 방향은 topology별 정책/배치 template을 재사용하고, 큰 도로망에서 연결부의 coverage 중첩과 간섭을 보정하는 것입니다.
지역별 최적 배치를 단순 합친다고 도시 전체 최소 LiDAR 수가 보장되지는 않습니다.
전역 budget/연결부를 다루는 별도 구성 최적화가 필요하며 현재 구현에는 없습니다.

## 17. 의존성과 License

주요 Python package:

```text
numpy, scipy, torch, open3d, gurobipy, pulp
pandas, matplotlib, PyYAML, plotly, python-pptx
```

`requirements.txt`를 사용하되 현재 완전한 버전 lock은 아닙니다.
실제 실행 버전은 결과 packages.txt, hardware.json을 기준으로 확인합니다.
PDF 변환은 LibreOffice, 한글 발표 글꼴은 Noto Sans CJK KR을 사용합니다.
Gurobi 라이선스는 별도 설치/등록이 필요하며 키/라이선스 파일은 저장소에 넣지 마세요.
300초 제한은 코드 설정이지 학교 라이선스가 강제로 정한 시간이 아닙니다.
`soft_mem_limit_gb`는 머신 전체 worker 합산 메모리를 제한하지 않습니다.

이 저장소는 오픈소스가 아닙니다. 코드/설정/데이터/문서/결과의 연구·논문·상업적 이용 조건은 [LICENSE](LICENSE)를 확인하세요.
