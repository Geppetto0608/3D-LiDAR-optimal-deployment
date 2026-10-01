# 공통 배치 엔진: 1차 구현

기존 Roadside CLI와 MAPPO v3를 유지하면서 두 application이 사용할 공통
문제 표현과 GREEDY/MILP 엔진을 추가했다. 로컬 UI는 구현됐으며 지도/가중치 조건부 RL은 아직 미구현이다.

## 로컬 배치 UI

```bash
.venv/bin/python placement_ui.py --port 8765
```

브라우저에서 `http://127.0.0.1:8765`를 연다. SSH 사용 시 클라이언트에서
`ssh -L 8765:127.0.0.1:8765 ai-server`를 실행하고 로컬 브라우저로 연다.
Box/도로 예제 또는 사용자 Mesh JSON을 불러온 뒤 K와 목적 가중치를 조절한다.
XY 사각형 영역의 target 중요도를 배수로 조정할 수 있다. GREEDY,
MILP(제한시간/gap 설정), 후보 선택형 MAPPO 새 학습, 회전 가능한 3D, 결과 JSON 다운로드를 지원한다.
화면은 단일 사용자용이며 서버 메모리에 현재 장면 하나를 유지한다.
동일 장면의 가중치를 변경할 때 raycasting을 다시 하지 않는다.

입력 예제: `configs/mesh_example.json`. `vertices[N,3]`, `triangles[F,3]`,
`poses[M,5]`, `targets[T,3]`, `importance[T]`, `costs[M]`, `sensor`를 포함한다.
좌표 단위는 m, pose는 x/y/z/yaw/pitch(deg)이고 센서 원점은 mesh 외부의
사용자가 승인한 설치 위치여야 한다. STL/OBJ의 자동 후보 생성, OSM 변환은 미구현이다.
도로 UI는 XY=10m, Z=3m, yaw=55deg, pitch=20deg의 별도 demo 격자이며 공식 실험과 다르다.
배치 결과에 입력 조건, 가중치, solver metadata를 저장한다. 원본 mesh JSON도 함께 보관해야 재현 가능하다.
중요도/penalty는 target 수 기반이다. 최대 연속 사각지대와 localization 성능은 아직 평가하지 않는다.

검증: `.venv/bin/python tests/test_placement_ui.py`

### 알고리즘 / K 비교

체크박스로 알고리즘을 복수 선택하고 단일 K 또는 K 범위를 지정한다.
3개 알고리즘 × K=3~5는 같은 가중치의 9개 독립 작업이며 MAPPO는 각 작업마다 새로 학습한다.
모든 실행은 버튼으로 시작한다. 비교표/그래프는 coverage, 중복 target 비율, 미관측 비율/수,
가중 목적값을 표시한다. 중복은 density>=2인 target 수이며 기존 overlap penalty 합계와 다르다.
미관측은 target 기준으로 연속 공간 면적이나 최대 사각지대 크기가 아니다.
실제 설치 수를 예산 K와 구분한다. 표/그래프에서 결과를 선택해 3D 배치를 전환한다.
결과 그래프에 알고리즘/K 묶음 체크박스와 개별 결과 체크박스를 추가했다.
임의의 알고리즘/K 조합을 동시에 선택할 수 있으며 재계산 없이 표시만 바뀐다.
기본 묶음 막대그래프와 K별 선 그래프를 전환한다. 3D 단일 선택과는 독립적이다.
각 실행 파라미터는 GREEDY 선택/MILP 풀이/MAPPO 학습 제목으로 구분한다.
실패는 누락점/오류로 남기고 다음 조합을 실행한다. 중단 버튼은 현재 작업 후 남은 작업을 생략한다.
전체 JSON(`placement_ui_comparison_v1`)에 공통 설정, 성공 결과, 실패/미실행 상태를 저장한다.
브라우저를 닫으면 현재 서버 작업만 계속되며 나머지 순차 작업은 시작되지 않는다.
`static/workbench.js`가 비교 실행과 시각화를 담당한다.

브라우저 검증 (Playwright Chromium, Gurobi 라이선스 필요):
`.venv/bin/python tests/test_placement_comparison.py`

### 실행량 조절과 MAPPO

- GREEDY: 최대 선택 단계 0~100, 실제 상한 min(K, 단계 수). 학습 횟수가 아니다.
- MILP: 제한시간 1~3600초, relative gap 0~1. 기본 30초/0.05, 1 thread.
- MAPPO: 32~20000 joint steps, seed 지정. 기본 1000 steps, CPU, 매번 새 신경망/optimizer.
- 가중치 수정은 기존 결과를 무효화한다. MAPPO/MILP는 버튼으로 명시적으로 다시 실행한다.
- `src/methods/candidate_mappo.py`는 기존 SharedMAPPO PPO 엔진을 재사용하는 고정 장면용 1-step joint candidate selection 모델이다.
- agent 입력은 ID, 행동은 후보 ID 또는 미설치. 중복 제안은 한 대만 설치한다. 보상은 공통 가중 목적값의 정규화 값이다.
- 기존 도로용 11-action MAPPO v3의 환경/정책과 다르다. 지도/가중치 조건부 전이 정책은 여전히 미구현이다.
- 표시 배치는 빈 배치를 포함한 탐색 최고 목적값이다. 최종 deterministic 정책 성능은 별도 metadata/화면 값이다.
- JSON에 학습 단계, seed, update 수, 학습 기록, 정책 평가를 포함한다. checkpoint 저장/재개/취소는 미구현이다.
- 짧은 테스트는 PPO 파라미터 갱신, 동일 seed 새 초기화의 재현성, 공통 evaluator 일치, 변경 가중치 적용을 검사한다. 성능 수렴 검증은 아니다.

## 구성

- `src/core/candidate.py`: 기존 Candidate/CandidateSet, legacy pose5 순서 유지.
- `src/core/models.py`: SensorPose, PlacementProblem, PlacementResult.
- `src/core/objective.py`: coverage, importance, overlap, count, cost 계수.
- `src/core/evaluation.py`: 선택 결과의 공통 목적값과 비가중 coverage.
- `src/core/visibility.py`: 임의 pose의 mesh raycasting API, 기존 광학 convention 사용.
- `src/roadside/candidate_generator.py`: 기존 순서와 pruning을 보존한 후보 생성.
- `src/roadside/problem_builder.py`: Roadside에서 공통 문제로 변환.
- `src/roadside/visibility_adapter.py`: 기존 cache/FOV를 common mask API로 노출.
- `src/robot/pipeline.py`: box 또는 mesh + 사용자 승인 mount pose + task target.
- `src/methods/greedy.py`, `milp.py`: 두 application이 공유하는 `solve_problem`.

## 실행

```bash
.venv/bin/python main.py robot --config configs/robot.yaml --dry-run
.venv/bin/python main.py robot --config configs/robot.yaml --result-json results/robot_demo/result.json
.venv/bin/python main.py roadside --config configs/experiment.yaml --dry-run
.venv/bin/python tests/test_common_placement.py --roadside-parity
```

`roadside`는 기본 scenario 0, K=1, GREEDY이다. YAML의 `roadside.scenario_id`,
`optimizer.method`, `optimizer.budget_k`로 명시할 수 있다. 기존 `milp/greedy/mappo`
명령은 기존 설정과 출력 형식을 유지한다. 새 application 명령은 `placement_v1`
JSON을 사용하므로 기존 결과 viewer의 입력으로 사용할 수 없다.

## 목적값

J = coverage_weight * covered_count
  + importance_weight * sum(covered target importance)
  - overlap_penalty * sum(max(observation_count - 1, 0))
  - sensor_count_penalty * selected_count
  - installation_cost_weight * sum(selected installation costs)

모든 계수는 유한한 비음수다. coverage_weight=1, 나머지=0이면 기존 목적값이다.
단위는 target 수 기반이며 UI용 정규화는 아직 적용하지 않았다.
target_importance는 target 순서의 배열, installation_costs는 candidate 순서의 배열이다.
Robot은 `robot.target_weights`, `robot.installation_costs`를 사용한다.
가중 Roadside 실행은 기존 top-M 및 coverage-only dedup을 해제하므로 계산량이 커진다.
이 모드에서는 후보를 한 번 생성한 후 PlacementProblem의 가중치를 바꿔 재사용할 수 있다.

## 구현 범위와 후속 단계

- Robot 예제는 실제 휠체어 성능 실험이 아닌 raycasting/optimizer smoke다.
- Robot 좌표는 metres, right-handed, pose 순서는 x/y/z/yaw/pitch(degrees)다.
- nonzero roll 광학 계산, mesh mounting surface 자동 추출, 센서 하우징 간섭 검사는 미구현이다.
- 승인된 외부 mount pose를 사용자가 제공한다. mesh 원점/단위는 입력 전에 맞춰야 한다.
- 사용자 mesh JSON과 영역 슬라이더 UI는 구현됐다. OSM 변환과 자유형 영역 painting은 후속 구현이다.
- robustness/연속 사각지대는 정의와 solver 지원이 완료되기 전 활성화하지 않는다.
- Robot localization/회피 성능은 이 geometric coverage로 검증되지 않는다.
- 새로운 지도와 가중치를 입력받는 RL은 별도 observation/reward 설계 및 학습이 필요하다.
- 기존 MAPPO의 reward와 checkpoint는 이번 단계에서 변경하지 않는다.
- 기존 cache key와 roadside FOV의 forward>0.5 조건도 유지한다. camera 실측 모델이 아니다.

## 회귀 기준

이번 변경 직전 실행한 Scenario 0 / K=1 / `configs/experiment.yaml` 기준:
target 7,200개, 후보 7,985개, 선택 ID 2427,
pose `[30.25,44.25,3.0,85.0,-20.0]`, covered_count 4,205.
전체 순서대로 결합한 packed candidate mask SHA256은
`a4e017d77f79700e477071ce1b41f4e05e70a3a1713156669565d47875ed333d`이다.
`test_roadside_golden`은 위 설정 전용 테스트이며 실험 설정을 변경하면 기준도 검토해야 한다.
