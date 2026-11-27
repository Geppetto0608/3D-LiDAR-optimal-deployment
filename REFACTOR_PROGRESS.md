# 공통 배치 엔진: 1차 구현

기존 Roadside CLI와 MAPPO v3를 유지하면서 두 application이 사용할 공통
문제 표현과 GREEDY/MILP 엔진을 추가했다. 로컬 UI는 구현됐으며 지도/가중치 조건부 RL은 아직 미구현이다.

## Prototype runtime 실험 추가 (2026-10-06)

### Fresh full-grid packed 기준선 완료 및 집계 (2026-10-14 확인)

- `src/roadside/packed_candidates.py`: full installable grid, 전체 orientation, pruning/dedup 없이 NPY memmap packed 저장.
- `prototype_full_grid.py`: 새 cache/후보 → packed GREEDY → swap 30초 → cut-MILP 300초 → direct 검산/3D 저장.
- 지도별 frozen-source 순차 worker, RAM96GiB/디스크 headroom 사전 검사, Gurobi 1 thread/90GB soft memory.
- 기존 online pose/기준선 archive 미사용. 4개 지도/K3,4 총8개, 새 root `results/prototype_full_grid_20261012`.
- 2026-10-12 20:41~22:04:28 KST 8/8 완료, 경과 5,003.48초. 실행 프로세스 종료 확인.
- 모두 status9/time limit, 상대 gap0.061~6.064%, discrete_optimal=false. direct 검산 기록 8/8 통과.
- 지도별 worker peak RSS 5.00~6.81GiB, 후보 mask 작성349.57~599.10초. 공유 서버 단일 측정이라 통제된 speedup 주장은 하지 않음.
- 테스트: 새5개(작은 실제 full-grid worker 포함), cut4개, pipeline7개, 문법/dry-run 통과.
- `prototype_full_grid_report.py`: 실제 archive pose 포함, 동일 target/density/coverage/K 검사. 기준선8개+온라인72개 통과.
- `comparison_report/comparison.csv` 18행, raw72행, validation summary 생성. 원자료/solver 변경 없이 집계.
- LS/MAPPO 평균은97.642/97.087%, 새 incumbent 대비 평균차이 -1.229/-1.784%p. LS가 모든18조건에서 MAPPO 평균보다 높음.
- LS4조건/MAPPO평균3조건이 이번 incumbent보다 높음. 전체 격자 해5/8은 이전 대표 격자 해보다 낮음. 최적성 주장이 아니라 제한시간 기준선 보강 필요.
- 다음: 검증된 이전 오프라인 feasible 해를 초기해로 보존, 반복 cold-start 계측, 독립 prototype 선택/holdout 평가.

### 공통 설치 격자 옵션과 영역 검사 (2026-10-12)

- `candidate.xy_domain=installable_grid` 추가. 생략 시 기존 bin_representatives/후보 순서 유지.
- 환경의 `installation_base_positions`에 기존 대표 격자와 전체 설치 가능 0.5m 격자 정의를 통합.
- 공통 `validate_installation_pose`를 국소 MAPPO/LS pose 검사에서 사용. 기존 이동 2m/반경 10m 유지.
- prototype_offline CLI에 명시적 `--xy-domain` 추가. 기본 config와 기존 결과/캐시는 수정하지 않음.
- `prototype_domain_audit.py`는 geometry로 먼저 영역 생성 후 저장 pose를 검사. raycast/학습/solve 없음.
- `results/prototype_domain_audit_20261012/report.json`: 72/72 포함, 새 영역 밖 0건, 기존 대표 XYZ 밖 8건 재확인.
- 4개 지도 dedup 전 pose 약 93만~119만 개. 이후 위 full-grid runner로 전체 visibility/MILP 재계산8개 완료.
- 5개 영역 테스트, pipeline 7개, 문법 검사, 기본 3방법 및 새 offline dry-run 통과.
- runtime 8개와 refinement study 5개도 통과(총 25개). 테스트용 소규모 계산만 실행.
- 기존 solver bound는 확장된 공간에서 무효. 물리 허용 공간 포함과 온라인 제한 탐색의 도달 가능성을 구분.
- 작업 중 프로젝트가 `/home/robotserver/hj_ws_main`으로 이동됨. 과거 결과의 절대 경로는 보존.

### Packed coverage-cut MILP 진단 (2026-10-12)

- 기존 8개 기준선 저장 완료. 7개는 presolve에서 시간 초과하여 GREEDY 초기해만 남음.
- `src/methods/coverage_cut_milp.py`: 같은 archive의 unweighted at-most-K maximum coverage를 lazy cut으로 표현.
- 처음부터 전체 incidence 행렬을 넣지 않음. 정수해마다 실제 union을 계산하고 위반한 제약만 추가.
- 전체 후보 교환 탐색 30초 + solver 300초. 온라인 배치를 warm start로 쓰지 않음.
- `prototype_milp_diagnostic.py`: archive/source/이전 해 검증, 새 해 direct raycasting, JSON/NPY/3D 저장.
- `prototype_cut_study.py`: frozen source에서 4개 target/K=3,4 순차 진단, 완료 대표 run 재사용 및 요약 CSV.
- test_coverage_cut_milp 4 tests(9개 실제 solver/brute-force 비교 포함), 기존 pipeline 7 tests 통과.
- 기존 geometry, candidate archive, 기본 MILP 함수/CLI는 수정하지 않음. 후보 영역 불일치 문제는 남아 있음.
- cache 재사용 진단 시간이므로 cold-map speedup으로 해석 금지. 현재 상태는 각 결과 manifest 참조.
- 대표 420002/K3 완료: 91.283% → 교환 탐색 99.918%, MILP 종료도 99.918%, solver upper bound 19547.4508 / gap 0.0791%.
- 교환 탐색 7.72초, solver 300.01초(time limit). 최적성 증명은 아니며 direct raycast 검산 통과.
- `results/prototype_cut_study_20261012`은 대표 run 1개 재사용 + 7개 순차 계산으로 2026-10-12 13:43:56 KST에 8/8 완료.
- 2개 이산 최적성 확인(100% coverage), 6개 time limit이나 유한 상한 확보. 최대 상대 gap 1.4774%.
- 저장된 direct-raycast 검증 8/8 참. 문서 갱신 시 학습/solve/raycast를 재실행하지 않음.
- 6개는 교환 탐색과 최종 objective가 같고, solver 추가 개선은 420002/K4와 420004/K4에만 있음.
- 최신 `online_comparison.csv`: paired 초기값 18개를 새 기준선에 연결. 이전 비교 CSV와 원자료는 보존.
- LS/MAPPO 평균 기준선 차이 -1.299/-1.854%p. 영역 불일치 포함 기술통계이며 최적성 gap이나 speedup 증거가 아님.
- README에 전체 8개 표, 비교 CSV 재생성 명령, 후보 공간/시간 범위 제한과 후속 계획 갱신.
- 집계 tests/test_cut_study.py 1개 추가 통과. 기존 결과/캐시는 보존.

### 새 지도 full-orientation 기준선 (2026-10-12)

- `prototype_target_baseline.py`: 완료된 paired study의 4개 target/K=3,4를 새 cache로 독립 MILP 실행.
- 온라인 snapshot의 기존 offline runner를 그대로 사용. top-M=0, 기존 2m 대표 XY, 300초/gap0/90GB/1 thread.
- 온라인 해 warm start 금지, target별 순차 실행, 가용 RAM 96GiB 미만이면 새 작업 시작 거부.
- source/온라인 evidence hash 고정, target 좌표 일치 검사, 상태·gap·bound와 시간 범위 분리 저장.
- 후보 영역 검사에서 72개 온라인 결과 중 8개(LS 1, MAPPO 7)가 기존 MILP 대표 XYZ 밖임을 확인.
  `online_domain.json`/CSV flags로 표시하며 global upper bound나 엄밀한 optimality gap으로 해석하지 않음.
- `results/prototype_target_full_20261012` 10월 12일 03:29:20 KST 8/8 완료. 7개는 유한 bound 없는 시간 초과 결과.
- 신규 tests/test_target_baseline.py 5개 + pipeline 7개 통과, dry-run 및 문법 검사 통과.

### 초기값별 paired refinement (2026-10-11)

- `prototype_refinement_study.py`: 최신 source/target library를 읽어 설치선별 센서 배분을 진단.
- 24개 target/K 중 22개는 branch/side별 배분 불일치, 가능한 보간 2개. 실패를 수치 0으로 넣지 않음.
- 먼저 K=3,4의 유효 초기값 18개: LS 18회 + MAPPO seed 0/1/2 총 54회 = 72회 비교.
- 초기 coverage를 보고 방법 하나를 고르지 않고 low/high/interpolation 각각 독립 보정.
- 같은 초기 pose hash, 반경 10m/soft limit 60초. 실제시간, overrun, ray 수, step 수를 구분.
- fresh CPU worker / 1 thread. 코드·source geometry/config/pose snapshot에서 실행.
- target full cache 금지, 보정 중 on-demand 질의만 사용. 별도 final direct 검산/3D 저장.
- `tests/test_refinement_study.py` 5 tests, 기존 pipeline 7 tests/runtime 8 tests 통과.
- 18개 초기값 평균: 초기95.661%, LS97.642%/3.91초, MAPPO97.087%/61.18초. 모든 조건에서 LS가 MAPPO 평균보다 높음.
- target full-grid 기준선 비교표와72개 pose 포함/density 검증 완료. 통제된 반복 cold-start 시간 비교는 TBD. 재현 명령은 README 참조.
- `results/prototype_refinement_smoke8_20261011` 6/6 완료, MAPPO 3회 각각 8 steps/PPO 1 update 확인.
- `results/prototype_refinement_paired_20261011` 한국시간 10월 12일 00:58:58에 72/72 완료, 실패 0건.
- 기존 geometry, MILP, MAPPO 알고리즘 및 라이브러리는 변경하지 않음.

### 라이브러리 형상 확장 경로 (2026-10-08)

- `prototype_offline.py --scenario-json`으로 명시적 synthetic 교차점 geometry 입력 지원.
- `configs/library_expansion.json`: source 4개(T/C 각각 폭 7/14m), 별도 target 4개(폭 9/12m 및 교차점 변경).
- `prototype_library.py`: 전체 orientation 후보 archive, K=1~6 MILP, 가족별 새 지도 전이/보간을 순차 처리.
- 기존 library는 보존. `results/prototype_library_shapes_20261008`은 10월 8일 20:38:37 KST 완료(약 2시간 35분).
- source 4개/K별 배치 24개 저장. analytic 인증 9개, time-limit best_known 15개(유한 bound/gap 없음).
- 순차 1 thread, MILP 300초/gap 0/soft memory 90GB. 목표 gap 0과 실제 최적성 인증을 구분.
- target 전체 cache와 MAPPO/target MILP는 실행하지 않음. source pose/geometry만 이용해 구간 비율 보간 후 direct raycast.
- 설치선/실제 센서 수 대응 불가는 unavailable, 투영 실패는 failed로 저장. 성공 결과로 바꾸지 않음.
- target/K 24개에서 단독 전이 48개, 보간 2개 완료. 보간 22개는 설치선 대응 불가, 실행 실패 0.
- 성공한 보간은 새 십자 420003/420004의 K=4뿐. 전체 대응 가능성이나 near-optimal 성능을 입증한 결과가 아님.
- 10월 9일 문서 확인: 50개 pose/coverage 배열과 JSON 일치, 3D/참조 파일 누락 0. 새 raycast 재검산은 아님.
- 다음 단계는 대응 가능한 보조 library 설계 검토, 그 뒤 국소 MAPPO 및 동일 target의 전체 후보 MILP 비교.
- tests/test_library_expansion.py 6 tests: 입력 분리/검증, dry-run, 불가 결과, 축소 실제 생성-전이 경로.
- 정확한 명령과 신규 geometry, 출력 구조는 README의 라이브러리 형상 확장 절 참조.

### 간단한 비대칭 형상 4개 추가 (2026-10-07)

- OSM 확장은 보류. `configs/shape_pilot.json`에 교차점 위치를 바꾼 T자 2개/십자 2개 정의.
- `Scenario.raw.geometry`를 공통 environment에서 처리하고 mesh/target/설치 판정/좌표/hash에 반영.
- 기존 중앙 맵 및 cache key 회귀 유지. 기존 scenario 파일과 experiment.yaml은 그대로.
- `prototype_runtime.py --shape-config configs/shape_pilot.json`으로 실행하며 width-only 경로도 유지.
- 4개 맵 x 선택/local-search/MAPPO = 12개 CPU worker 완료. K=3, seed=0, MAPPO 256 steps.
- `results/prototype_shape_pilot_20261007/index.html`에 맵과 3D 링크, CSV와 상세 metadata 저장.
- 새로운 형상에서도 국소 보정은 작동하지만 1 PPO update 파일럿이므로 수렴/우월성 주장은 불가.
- 이 파일럿 이후 2026-10-08에 같은 맵의 GREEDY/MILP top-10 기준선도 완료. 아래 기록 참조.

### 같은 형상의 PPO 학습량 확장 (2026-10-07)

- `results/prototype_shape_ppo4096_20261007/`: 4개 맵, K=3, seed=0, 동일 초기 배치/반경/CPU.
- joint steps 256 -> 4,096, 업데이트 1 -> 16. rollout/epoch/기본 YAML은 변경하지 않음.
- 안전 시간 제한 600초, 전부 조기 종료 없이 완료. 기존 정책을 재사용하지 않음.
- 최고 coverage: t_left 96.572%, t_long_stem 89.385%, cross_left 81.405%, cross_asymmetric 78.401%.
- online wall time 약 89~96초/맵. direct raycast 재검산 4/4 통과.
- 마지막 비대칭 십자 정책은 69.477%로 최고 배치보다 낮음. 수렴/효율 우위 주장은 불가.
- README에 정확한 실행/검증 명령과 이전 256-step 결과 비교 기록. 기존 결과 보존.

### 형상 기준선 완료 (2026-10-08)

- `results/prototype_shape_baselines_20261008/`: K=3, GREEDY/MILP 각 4개, 총 8개 완료 및 direct 재검산 8/8 통과.
- MILP: top-10, 300초, gap 5%, 1 thread, soft memory 26GB. 기존 YAML/알고리즘 변경 없음.
- MILP coverage: t_left 99.905%, t_long_stem 99.845%, cross_left 92.975%, cross_asymmetric 97.660%.
- 모두 status 2. cross_left gap 1.053%, 나머지 solver-reported gap 0%. 후보 집합 안의 인증만 해당.
- 중단 전 완료 5개 보존, 남은 3개 재실행. 중단 기록은 `interrupted_attempts/`, 재시도는 `resume_history` 보존.
- `shape_baseline_report.py`: 기존 select/local-search/MAPPO 256/4096 + 기준선을 검증·집계하여 24행 CSV/PNG/PDF/HTML 생성.
- 같은 geometry/target/LiDAR/K 확인. 초기화/전처리 포함 runtime과 solver-only 시간 구분.
- MAPPO 4096은 네 맵 모두 MILP보다 낮고, GREEDY보다 높은 것은 한 맵뿐. 효율/near-optimal 주장 보류.
- 정확한 실행/재검산/집계 명령과 전체 수치는 README의 2026-10-08 기준선 절에 기록.

### 기존 순차 runtime 경로

- `prototype_runtime.py`: 별도 순차 runner. 새 process/새 cache, 전체 online 시간과 단계별 시간,
  ray 수·평가 수·메모리 계측, 입력 및 source snapshot, CSV 저장. `--execute` 없으면 계획만 확인.
- `prototype_segments.py`: 동일 branch/side 설치 구간의 비율 좌표 보간. 기존 거리 보간은 유지.
- `src/methods/local_search.py`: MAPPO와 같은 action/mask/radius를 사용하는 단일 센서 이웃 탐색.
- 단독 전이 두 개와 보간 중 coverage가 가장 높은 초기 배치 하나만 골라 MAPPO 또는 local-search로 보정.
- `src/core/runtime.py`: opt-in 계측. 기존 물리 모델·설정 기본값·기존 결과는 유지.
- 실행 명령·수식·측정 범위는 README의 `순차 Runtime 비교 경로` 절에 기록.
- 정식 runtime 성능 비교는 TBD. 경량 통합 실행은 학습 수렴이나 속도 우위를 입증하지 않음.

## 로컬 UI 실행

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
