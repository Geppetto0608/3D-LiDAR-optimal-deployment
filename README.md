# 3D LiDAR Optimal Deployment

**연구 질문 (2026-11-18 방향 정리):** A: 전체 topology 배치 library와 B: 구성 요소·연결부
배치 library를 새 도로에 전이·조합하여, 처음부터 계산하는 것보다 적은 비용으로 좋은 배치를
구할 수 있는가? 이후 대형 도로망의 전역 센서 배분·목표 coverage 필요 대수와 제약부 재최적화로 확장합니다.
GREEDY·MILP·국소 탐색(LS)·MAPPO 및 geometry-only 선별을 비교하며,
library 재사용의 일관된 이점·큰 도로망 구현·동일 품질 시간 절감·최소 대수 인증은 아직 검증 과제입니다.

**보류한 학습 확장:** library에 저장된 geometry·복수 배치·관측 정보를 학습하여 source 검색,
배치 후보 생성 또는 성능 예측에 활용하는 방법도 후속 후보로 유지합니다. 우선은 비학습 A/B
전이·조합을 검증하며, library 사전학습과 새 지도마다 수행하는 fresh MAPPO 보정은 구분합니다.
모델·학습 실행은 미정/미구현입니다. 상세 범위와 검증 조건은 [연구 계획](TITS_RESEARCH_PLAN.md)에 기록합니다.

**A/B 오프라인 저장 기준:** [LIBRARY_STORAGE_PROTOCOL.md](LIBRARY_STORAGE_PROTOCOL.md)에
K별 최대 union coverage의 문제 정의, 최고 feasible 배치 보존, 상한/gap, 대체 배치와 최소 대수
판정을 고정했습니다. 두 트랙은 공통 형식을 사용하며 source 생성과 새 지도 전이 실험은 구분합니다.

**2026-11-27 통합 정리:** [실험환경·계산 수식·A/B 알고리즘·전체 이력·TR-C 준비도](RESEARCH_STATUS_20261127.md).
현재는 library 재사용의 개발용 제약부 비교와 결과 검수까지 완료한 단계입니다. 미사용 도로 평가,
통제된 품질/시간 비교·구축비 상각, 임의 도로망의 필요 센서 수 검증은 남아 있습니다.
아래 날짜별 기록의 과거 '현재 방향'은 당시 상태이며, 최신 범위는 이 통합 문서를 먼저 봅니다.

**GitHub에서 읽는 순서와 결과:** [연구 인덱스 및 versioned 요약 CSV](docs/research/README.md).
최신 검수 결과의 작은 표와 provenance는 저장소에 포함했습니다. `results/`의 원본 pose/mask,
대용량 로그·3D 자료는 여전히 제외되며, 아래 해당 경로 링크는 로컬 결과용입니다.

## Library 이점 원인 분리: 고정 Q 실험 추가 (2026-11-27)

[실험 명세](LIBRARY_MECHANISM_PROTOCOL.md), [고정 설정](configs/library_mechanism_20261127.json),
[runner](library_mechanism_study.py)를 추가했습니다. **정식 성능 실험은 아직 실행하지 않았습니다.**
기존 source 24개와 개발 지도 8개, 저센서·비용·중요 구간 제약은 유지합니다.

- source **1/3/6개**, profile/K별 **최고 배치만/최대 12개 대체 배치**를 분리 비교합니다.
- 근접 source 검색과 무작위 source, prefix/diverse, geometry union/portfolio, random을 비교합니다.
- 모든 방법의 실제 조회 수를 **Q=256으로 고정**합니다. library 후보가 부족하면 geometry union으로
  보충하고 library/보충 후보 수를 별도 기록합니다. **순수 library가 아닌 명시적 hybrid 실험**입니다.
- Q는 pose 수입니다. XYZ 재사용으로 ray 수와 계산시간은 달라질 수 있어 각각 기록합니다.
- 동일 query set 여부, feasible 비율, coverage/중첩/비용/시간과 source 수별 차이를 저장합니다.
  seed 평균을 조건 안에서 먼저 구하며, 불가능/미확정 결과를 coverage=0으로 바꾸지 않습니다.
- 정식 계획은 **16 cases, 온라인 2,760행, 전체 후보 기준선 184개**입니다. RL/검출은 추가하지 않습니다.
- 새 geometry 8개는 config의 `split.reserved`로 예약했습니다. 이번 runner에서는 사용을 거부합니다.
  이는 향후 평가용 분리이며, 미사용 topology나 실제 도로 일반화가 검증됐다는 뜻은 아닙니다.

계획만 확인하는 명령(학습·solve·raycast 없음):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_mechanism_study.py \
  --spec configs/library_mechanism_20261127.json
```

실제 실행·smoke 명령과 인증/시간 해석은 명세에 있습니다. 정식 실행은 시작 전 디스크 약 16GiB
이상을 요구합니다. 새 실험 수치를 기존 완료 결과에 섞지 않으며, 검증 상태는 명세 하단에 기록합니다.
**구현 검증 완료:** 회귀 135 tests, 문법 검사와 새/기존 dry-run 통과.
`results/library_mechanism_smoke_20261127/`의 A/B 2 cases·88행에서 Q=32 고정, 제약 검사,
저장 mask/직접 raycast 일치를 확인했습니다. 짧은 smoke이며 성능 우위·최적성의 근거는 아닙니다.

## 저센서·설치비용·중요 구간 제약 실험 (2026-11-24)

`library_budget_study.py`를 추가했습니다. 앞선 **검증 완료 source library 24개를 그대로 고정**하고,
센서 수를 줄인 뒤 새 지도에서 비용/중요 구간 제약을 만족하는 배치를 선택합니다.
기존 library 및 완료 결과는 덮어쓰지 않습니다. **2026-11-24 18:35:24~11-25 17:56:52 KST에
2 worker로 16/16 cases를 완료했습니다.** 전체 경과 시간은 84,088.43초(약 23시간 21분 28초),
실패 worker는 0개입니다. [manifest](results/library_lowk_cost_20261124/manifest.json)와
아래 2026-11-27 완료 결과 검수를 확인합니다. 이 실행은 학습이 아닌 제약부 배치 최적화입니다.
상세 [실험 명세](LIBRARY_LOWK_COST_PROTOCOL.md)와 [고정 설정](configs/library_lowk_cost_20261124.json)을 저장했습니다.

| 항목 | 이번 범위 |
|---|---|
| A 지도 | 120/180m T자·십자 4개, K=2,3,4 |
| B 지도 | 160/240/320m 연결 도로 K=3,4,6; 480m는 K=6,8 |
| 가림 | 기존 mixed/dense 정적 가림, 8개 geometry × 2단계 = 16 cases |
| 제약 profile | 기본 / 비용 / 비용+중요 구간 60% / 비용+중요 구간 80% |
| 공통 설치 제약 | 기존 금지 구간, pole당 1대, 최소 간격 4m, at-most-K |
| 조회 후보 | 요청 Q=256, library pool이 작으면 모든 방법에 동일 실제 Q 적용 |
| 비교군 | library prefix/diverse, geometry union/portfolio, random 3 seeds |
| 완료 산출물 | 온라인 1,288행(1,274 feasible / 14 infeasible), 전체 후보 기준선 184개(182 / 2) |

실제 가림을 반영한 visibility를 `a_ij`, 선택을 `x_i`, 중요 구간 target을 `T_c`로 두면:

```text
d_j = sum_i a_ij x_i,  y_j = 1[d_j >= 1]
maximize sum_j y_j
sum_i x_i <= K
sum_i c_i x_i <= 1.25 K                 # 비용 profile만
sum_(j in T_c) y_j >= ceil(tau |T_c|)    # tau=0 / 0.60 / 0.80
c_i = round(1 + 0.10 max(z_i-3,0) + 0.008 distance(xy_i,H), 6)
```

`H`는 geometry만으로 정한 service anchor의 집합이며 실제 전력/통신 시설 위치가 아닙니다.
비용은 **합성 상대 단위**, 거리도 가장 가까운 anchor까지의 직선거리 proxy입니다.
실제 견적·케이블 경로·통신 연결·시설 용량을 모델링하지 않습니다. 60%/80%도 운영 기준이 아닌
사전 고정한 민감도 실험값입니다. 불가능하다고 자동 완화하지 않으며 가중 penalty로 대신하지 않습니다.
중첩률은 전체/covered target 두 분모로 모두 보고하지만, 이번 단계에서는 필수 중첩 하한을 0으로
두어 비용과 중요 구간 단일 coverage 효과를 분리합니다. 앞선 필수 중첩 실험은 별도로 보존합니다.

각 방법은 실제 target raycast 전에 후보 계획을 고정하며, 동일 제약부 MILP로 최종 배치를 고릅니다.
후보 선별에 사용하는 geometry portfolio의 광학 surrogate에도 같은 비용/중요 구간 제약을 적용합니다.
전체 후보 기준선은 온라인 결과 확정 후 계산하고 온라인 해를 warm start로 읽지 않습니다.
최종 배치는 직접 raycast와 비용/중요 구간/설치 제약 재검산을 통과해야 저장 완료됩니다.
`infeasible`과 `no_feasible_found`는 구분하며 coverage 0으로 채우지 않습니다.
조회 후보군의 infeasible은 전체 지도에서의 불가능 증명이 아닙니다.

기준선 K당 1,800초, 온라인 선택 300초, Gurobi 1 thread이며 2 worker/solver soft memory 24GB입니다.
정체만으로 근사최적이라 하지 않고 유효한 상한 대비 gap≤1%를 함께 요구합니다.
solver 예산 합은 약 199.33 worker-hours이며 예상 완료 시간이 아닙니다. 전처리/모델 생성은 별도이고
전체 hard limit 168시간, worker당 48시간입니다. CPU 최적화이며 **MAPPO 학습이 아닙니다**.
기존에 사용한 개발 지도에서 제약을 추가하는 시험으로, 미사용 실도로 일반화나 여러 가림 상태에
공통으로 강건한 단일 배치를 구하는 실험도 아닙니다.

```bash
# --execute 없으면 읽기 전용 계획 확인
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_budget_study.py \
  --spec configs/library_lowk_cost_20261124.json

# 새 output root만 허용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_budget_study.py \
  --spec configs/library_lowk_cost_20261124.json \
  --output results/library_lowk_cost_20261124 --execute --detach
```

`source/`, `library/`, `library_provenance.json`, 설정/hash를 실행 폴더에 고정합니다.
case별 `installation_costs.npy`, `cost_model.json`, `query_plans.json`, frontier/시간곡선/관측 배열,
검산 결과를 저장하고 **모든 case가 완료된 후** root `comparison.csv`를 만듭니다.
진행 중에는 supervisor가 생존·종료·자원만 감시하며 성능 로그는 완료 후 분석합니다.

실행 전 검증: 비용/중요 구간/중첩의 작은 전수조사 대조를 포함한 **118 tests**, 문법 검사,
새 실험 계획 및 기존 MILP/GREEDY/MAPPO dry-run 3개 통과.
[최종 축소 실행](results/library_lowk_cost_smoke_20261124_v2/manifest.json)은
2/2 cases, 80/80행 및 선택 pose 직접 raycast 검산을 완료했습니다. 이는 성능 증거가 아닙니다.
첫 축소 실행의 성긴 설치 격자 때문에 발생한 설치 측면 누락은 수정하고 실패 기록은 보존했습니다.

### 완료 결과와 검수 (2026-11-27)

**같은 저센서·비용·중요 구간 제약에서 library 전이가 더 높은 평균 전체 coverage를 얻었습니다.**
모든 방법이 동일한 182/184조건에서 feasible이므로, library만 제약을 충족했다는 결과는 아닙니다.
아래 품질은 공통 feasible 조건 182개의 동일 가중치 평균이며 random은 조건별 3 seeds를 먼저 평균했습니다.
불가능한 2조건은 품질 평균에서 제외하고 별도로 보고합니다. 이를 coverage 0으로 채우지 않습니다.

| 방법 | 전체 coverage | 중요 구간 coverage | 중첩률 | 평균 온라인 시간 (해당 지도의 모든 K) |
|---|---:|---:|---:|---:|
| Library prefix | 78.815% | 88.687% | 7.142% | 58.12초 |
| Library diverse | **78.954%** | 89.247% | 7.090% | 94.08초 |
| Geometry union | 74.071% | 88.156% | 5.294% | **36.18초** |
| Geometry portfolio | 75.671% | **89.983%** | 6.270% | 101.75초 |
| Random (3-seed 평균) | 76.514% | 87.343% | 7.336% | 137.27초 |

중첩률은 전체 target 중 `density >= 2`인 비율이며 이번 목적함수나 필수 하한이 아닙니다.
비용은 상한 제약이지 최소화 목적이 아니며, 중요 구간 coverage도 최소 요구량이지 최대화 지표가 아닙니다.
따라서 library가 비용·중첩·중요 구간을 모두 더 좋게 만든다고 주장하지 않습니다.
기본 제약만으로 구한 배치를 그대로 비용 profile에 사용하면 방법별 약 82.6~89.1%가 비용 상한을
위반합니다. 이번 비용 제약은 실제로 배치 선택을 제한했으며 단순한 metadata 추가가 아닙니다.

시간은 **64개 case/profile의 전체 K frontier를 한 번씩** 집계한 평균입니다. 같은 시간이 K별
CSV 행에 반복되어도 중복 가중하지 않습니다. 지도에 따라 K가 2개 또는 3개이므로 단일 K의 latency가
아닙니다. 준비·후보 선별·조회·제약부 선택을 포함하며 source 구축, 프로세스 시작/import,
최종 독립 검산/export는 제외합니다. 2 worker 공유 서버의 단일 측정이므로 통제된 speedup은 아닙니다.

- Diverse vs portfolio: 평균 **+3.283%p**, 163승/0동률/19패입니다. 8개 geometry 각각의
  조건 평균 차이도 모두 양수이나 모든 개별 조건에서 우세하지는 않습니다.
- Diverse vs random: **+2.439%p**, 165승/1동률/16패입니다. Union 대비는 +4.883%p입니다.
- Diverse vs prefix: **+0.139%p**인 반면 전체 K 평균 시간은 **35.96초 더 깁니다**.
  간단한 prefix도 주요 비교군으로 유지해야 하며 diverse의 비용 대비 추가 이점은 작습니다.
- 실제 Q는 A=139/143/151/162, B=256입니다. 각 case의 모든 방법은 동일 Q를 사용합니다.
  A에서는 전이 pool 전체가 Q에 들어가므로 prefix/diverse의 조회 후보 집합이 같습니다.
  따라서 A에서 두 방식의 독립적인 선별 효과를 검증했다고 해석하지 않습니다.
- B480은 K 2개, 나머지는 3개이므로 위 표의 지도별 가중치는 다릅니다. 지도 8개를 동일 가중치로
  다시 평균한 coverage도 diverse 78.800%, portfolio 75.490%이며 별도 CSV에 저장했습니다.
  1,288행은 독립적인 1,288개 도로나 통계 검정용 표본 수가 아닙니다.

### 불가능 조건과 기준선 인증

온라인 infeasible 14행은 **B320m의 mixed/dense, K=3, cost_critical80** 두 조건에서
7개 방법/seed 조합 모두가 불가능한 경우입니다. 전체 후보 MILP도 두 조건을 infeasible로 종료했습니다.
별도 유한 상한 검산에서 중요 target 1,989개 중 필요한 1,592개(80%)에 대해 가능한 관측 수의
상한은 mixed **1,488개**, dense **1,434개**였습니다. 이는 다음 세 경우의 상한 중 최댓값입니다.

1. 둘 이상의 중요 연결부를 보는 pose가 포함된 경우: 단일 coverage 합을 이용한 상한.
2. 그런 pose가 없고 연결부 하나를 놓치는 경우: 나머지 연결부의 전체 target 수 상한.
3. 세 연결부에 한 대씩 설치하는 경우: 비용 3.75 이하의 cost/count 조합 전수 열거 상한.

3번에서는 pole/간격 제약을 완화했으므로 가능한 집합을 줄이지 않습니다. 그 조건부 상한
1,182/1,100개를 전체 문제의 상한으로 혼동하지 않습니다. 이 검증은 선언된 후보 격자와 합성 비용
모델에서의 불가능 증명이며, 연속 공간이나 실도로에서 3대가 불가능하다는 뜻은 아닙니다.

전체 후보 기준선의 feasible 182개 중 **109개 exact, exact 포함 143개 gap≤1%**입니다.
나머지 39개는 1%를 넘으며 최대 상대 gap은 **125.952%**입니다. 종료 상태는 exact/tolerance
status 2가 109개, 시간 제한 36개, gap+정체 조기 종료 34개, soft memory 제한 3개,
infeasible 2개입니다. 모든 기준선을 근사최적 정답으로 취급할 수 없습니다.
온라인 90행이 전체 후보 기준선의 저장 incumbent보다 높지만 상한을 넘지는 않았습니다.
이는 약한 incumbent를 개선한 것이며 전역 최적해를 능가한 결과가 아닙니다.

### 검수 자료와 재현

별도 [검수 보고서](results/library_lowk_cost_review_20261127_final/REPORT.md)와
[검증 기록](results/library_lowk_cost_review_20261127_final/validation_summary.json)에 범위를 기록했습니다.
실험 원자료를 수정하지 않고 `library_budget_report.py`가 다음을 확인합니다.

- source 24개 archive, 240개 K/profile 결과와 대체 배치 2,748개를 저장 mask로 재평가.
- 16개 geometry·비용 배열·도로 상대 전이를 재생성하고 448개 조회 계획을 재구성.
  시간 제한 optical portfolio는 저장된 surrogate 해의 유효성을 검사하며 재최적화하지 않음.
- 온라인 1,288행/기준선 184개의 상태, feasible pose·비용·간격·K·중요 구간 제약과 수치 대조.
- target 전체 후보 **40,840개**를 새 scene에서 직접 raycasting, 제출 ray **158,998,838개**.
  모든 저장 mask와 일치하고 8쌍의 dense visibility가 mixed의 부분집합임을 확인.
- source/input hash, root/case CSV, solver bound/gap/flag, 시간곡선 및 시간 집계 경계 일치 확인.
  검수 전후 원자료 hash 동일. 기존 대규모 solve/학습 재실행 없음.

이는 같은 Open3D/광학 모델의 재계산 검증이지 실제 센서 정확도 또는 별도 물리 모델의 검증이 아닙니다.
source 24개를 다시 raycasting하지는 않았고, feasible 대규모 MILP의 최적성을 별도 solver로 재증명하지도
않았습니다. 첫 검수의 단일 연결부 가정은 실제 다중 연결부 후보 36개를 발견해 보완했으며,
실패 기록은 `results/library_lowk_cost_review_20261127/failure.json`에 보존했습니다.
이는 검수 도구의 적용 조건 수정이며 실험 원자료 오류를 수정한 것은 아닙니다.

```bash
# 저장 결과 검수. 원본 밖의 새 output 필요, --direct는 전체 target 후보를 새 raycast로 대조
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_budget_report.py \
  --root results/library_lowk_cost_20261124 \
  --output results/library_lowk_cost_review_repeat --direct

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_budget_report.py
```

검수 도구 7개를 포함한 관련 **125 tests**, 문법 검사 및 기존 CLI dry-run 3개를 통과했습니다.
작은 테스트 solve는 포함되지만 대규모 실험 재실행은 아닙니다. 파생 CSV에는 방법/track/profile/scene/
geometry별 품질, 지도 동일 가중치 평균, K 중복을 제거한 시간, paired 차이 및 불가능 조건을 보존합니다.
결과 경로는 git ignored이므로 clone에는 포함되지 않습니다. 다음 우선순위는 약한 기준선의 보강과
미사용 geometry/장애물 seed 평가이며, 이번 표만으로 실도로 일반화·보편적 시간 절감·RL 필요성을
입증하지 않습니다. 앞선 고센서 실험과 K/제약이 달라 coverage 하락을 알고리즘 퇴보로 비교하지 않습니다.

## 대형 맵·필수 중첩·정적 가림 실험 (2026-11-23)

새 runner `library_rsu_stress.py`와 [실험 명세](LIBRARY_RSU_STRESS_PROTOCOL.md),
[고정 설정](configs/library_rsu_stress_20261123.json)을 추가했습니다.
**2026-11-23 19:22:10~11-24 16:09:59 KST에 완료했습니다.** 전체 경과 시간은
74,869.60초(약 20시간 48분)이며 source 24개, 평가 case 24개, profile 포함 48조건,
온라인 비교 2,016행을 완료했습니다. 실패 worker는 없습니다.
[실행 manifest](results/library_rsu_stress_20261123/manifest.json)와 아래 완료 분석을 확인합니다.
이 실험은 **제약부 배치 최적화**이며 MAPPO 학습이나 실제 차량 검출 실험이 아닙니다.

| 항목 | 새 실행 범위 |
|---|---|
| 오프라인 library | A 12개 + B 12개, 총 24 source; union K=1~8 및 중첩 제약 K=4,8 |
| 새 지도 | A 120/180m 정사각 T자·십자 4개; B 160~480m, 굽힘 15~75도 4개 |
| 가림·제약 | clear/mixed/dense × union/redundancy, 48조건 |
| 실제 target 후보 | 1,480~4,460 poses; 새 지도 설치 K=4,8,12 |
| 조회 예산 | 128/256 poses, library 부족 시 실제 수로 모든 비교군을 동일하게 제한 |
| 비교군 | library prefix/diverse, geometry union/portfolio, random 3 seeds |
| 전체 비교 | 2,016/2,016행 완료; 모두 feasible, 실패를 coverage 0으로 채우지 않음 |

공통 설치 제약은 금지 구간, pole당 한 대, 센서 간 최소 4m입니다. redundancy 조건에서는
**전체 target의 10% 이상 및 교차부/연결부 target의 30% 이상을 두 대 이상이 관측**해야 합니다.
값들은 운영 기준이 아닌 사전 고정한 시험 조건입니다. 차량·트럭·패널 크기의 불투명 정적
box와 기존 건물을 사용하며 실제 교통·검출 모델은 아닙니다. 모든 가림 단계의 target/후보 집합은
동일하게 유지합니다. A의 target은 0.5m XY/높이 3개, B는 1m station/lateral/높이 3개입니다.
설치 후보는 nominal 4m bin/station, 높이 3/5m, yaw 5개, pitch 2개입니다.
전체 0.5m 설치 공간을 최적화하거나 임의 도시 도로망을 구현한 것은 아닙니다.

source·전체 후보 기준선은 K당 최대 1,800초, 조회 후보군은 300초입니다.
정확한 인증 외의 조기 종료는 300초 이상·120초 개선 없음·상대 gap 1% 이하를 함께 요구합니다.
정체만으로 근사최적이라고 하지 않습니다. 중첩률, feasible 상태, 상한/gap, 시간 곡선 CSV,
전처리·후보 선별·raycast·선택 시간 및 ray 수를 저장하고 완료 후 비교합니다.
4 CPU workers이며 worker/메모리/디스크 안전장치와 최대 120시간 전체 상한을 둡니다.
solver 제한의 합은 직렬 360시간 예산으로, 4개 동시 실행 시 약 90시간에 해당하지만
이는 **예산 합계이지 예상 소요 시간은 아닙니다**. 쉬운 문제는 인증 후 일찍 끝납니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u library_rsu_stress.py \
  --spec configs/library_rsu_stress_20261123.json \
  --output results/library_rsu_stress_20261123 --execute --detach

tail -F results/library_rsu_stress_20261123/launcher.log
```

동일 output을 덮어쓰지 않습니다. 실행 시 `source/`, 설정/hash, package와 hardware 정보를
고정합니다. supervisor는 생존·자원·실패만 확인하며 중간 성능 분석은 하지 않습니다.
`manifest.json`이 최종 상태이며 완료 시 root `comparison.csv`를 생성합니다.
개별 K의 구조화된 경과 기록은 `library/<source>/<profile>/progress.jsonl`에 있습니다.
launcher 로그가 조용해도 긴 solver 실행 중일 수 있습니다. 자세한 수식·방법·시간 경계는
위 명세를 사용하며, 생성 결과는 ignored 경로라 clone에 포함되지 않습니다.

실행 전 검증: 신규 테스트 10개(작은 MILP 18문제 전수조사 대조 포함), 기존 pipeline 7개,
기존 CLI dry-run 3개 및 문법 검사 통과. 최종 축소 실행
[smoke v4](results/library_rsu_stress_smoke_20261123_v4/manifest.json)는 source 4개 + 평가 4개,
160행 완료 및 모든 작업의 독립 direct-raycast 검산을 통과했습니다. 최대 크기 두 지도에서도
전체 후보 mask 생성, 일부 pose 직접 검산, 짧은 제약부 solve 연결을 확인했습니다.
이는 실행 전 기능 검증 기록입니다. 실행 중 성능 로그를 분석하지 않았으며,
2026-11-24 완료 후 저장 CSV/JSON/배열을 이용해 아래 분석을 수행했습니다.

### 완료 결과와 성능 해석 (2026-11-24)

**이번 고정 시험에서는 library 기반 후보 전이가 geometry-only보다 높은 평균 coverage를
얻었습니다.** 실제 도로 일반화, 모든 조건의 근사최적성, 통제된 speedup의 증명은 아닙니다.
두 library 방식은 같은 source 후보군을 사용하고, prefix는 순서대로, diverse는 광학적
coverage/중첩 surrogate와 공간 분산을 이용해 조회 후보를 선별합니다. geometry union/portfolio와
random은 library를 사용하지 않습니다. 이후 실제 mask로 배치를 선택하는 제약부 MILP는 공통입니다.

| 방법 | 평균 coverage | 평균 중첩률 | 평균 온라인 시간 (K=4,8,12 전체) |
|---|---:|---:|---:|
| Library prefix | 93.038% | 47.055% | 147.12초 |
| Library diverse | **93.117%** | 46.737% | 182.60초 |
| Geometry union | 91.299% | 42.977% | 137.75초 |
| Geometry portfolio | 92.309% | 44.167% | 231.45초 |
| Random (3-seed 평균) | 92.283% | 44.792% | 236.92초 |

coverage/중첩은 8개 geometry × 3개 가림 × 2개 profile × 2개 조회 예산 × 3개 K의
288조건을 동일 가중치로 평균했습니다. Random은 각 조건에서 seed를 먼저 평균했습니다.
2,016행은 독립적인 2,016개 지도가 아닙니다. 중첩률은 `density >= 2`인 전체 target 비율이며,
중첩 자체를 최대화하는 목적함수가 아니므로 높다고 무조건 좋은 결과는 아닙니다.
시간은 96개 case/profile/Q에서 K=4,8,12를 모두 계산한 비용의 평균입니다. 단일 K의 latency가
아니며 source 구축, 프로세스 시작/import, 최종 독립 검산/export를 제외합니다.
4개 worker의 공유 서버 측정이므로 표의 시간 비율을 통제된 speedup으로 주장하지 않습니다.

- Diverse vs portfolio: 평균 **+0.808%p**, 167승/90동률/31패입니다. 8개 geometry 각각의
  조건 평균은 모두 양수지만 모든 개별 조건에서 이긴 것은 아닙니다.
- 가림별 diverse의 portfolio 대비 차이는 clear +0.354%p, mixed +0.722%p,
  dense +1.349%p입니다. 이번 고정 장애물 배치에서 더 큰 차이가 관찰됐다는 의미입니다.
- K=4에서 diverse/portfolio는 81.324%/79.852%, K=8은 98.145%/97.361%,
  K=12는 99.882%/99.713%입니다. 센서 예산이 큰 조건은 포화에 가까워 차이가 작습니다.
  K는 at-most 예산이며 K=12 결과가 모두 실제 12대를 설치했다는 뜻은 아닙니다.
- Diverse의 prefix 대비 개선은 평균 +0.079%p인 반면 기록된 시간은 35.48초 더 깁니다.
  현 조건에서 복잡한 후보 선별의 추가 이점은 작으며 기본 library 방식도 함께 보존해야 합니다.
- 속도 우위도 조건부입니다. B/redundancy에서 diverse는 평균 475.47초,
  portfolio는 384.11초로 library 쪽이 더 오래 걸렸습니다.

### 제약 충족과 최적성 인증

온라인 **2,016개 모두** 금지 구간, pole당 한 대, 최소 4m, K 제약을 만족했습니다.
이 중 redundancy 1,008개는 전체 중첩 10%/critical 중첩 30% 조건도 만족했습니다.
비교군 모두 가능한 배치를 찾았으므로 library만 제약을 충족할 수 있었다고 주장하지 않습니다.

전체 후보 기준선 144개 중 exact는 69개, 상대 gap 1% 이하는 exact 포함 110개입니다.
**21개는 메모리 제한, 19개는 시간 제한으로 종료**했고 최대 상대 gap은 **80.039%**입니다.
이는 worker 실패가 아니라 feasible 해를 반환한 solver 종료입니다. 12GB soft memory 제한으로
1,800초에 도달하기 전에 종료한 경우도 있습니다. source 240개 역시 exact 181개,
상대 gap 1% 이하 229개이므로 library 전체를 인증된 근사최적 해라고 부르지 않습니다.

평가값 C, 독립 전체 후보 기준선 feasible 값 L, 상한 U를 coverage %로 나타내면,
해당 이산 후보 공간의 최적값 C*에 대한 손실은 다음 구간으로 보고합니다.

```text
max(0, L - C) <= C* - C <= max(0, U - C)   [percentage points]
```

Library diverse의 평균 손실 상한은 union 조건 **0.303%p**, redundancy 조건 **3.736%p**입니다.
전자는 이번 이산 공간에서 작은 평균 손실을 뒷받침하지만 후자는 특히 K=4에서 기준선이 약합니다.
개별 조건 보장은 별도 CSV의 상한을 확인해야 합니다. 위 구간은 통계적 신뢰구간이 아니며,
solver 상대 gap `(U-L)/L`과도 다릅니다. 온라인 해가 기준선 incumbent보다 높아도
MILP 최적값을 능가한 것이 아닙니다. 중간 coverage 정체만으로 최적성을 인증하지 않습니다.

### 비용, 후보 확대 및 다음 확인

source 구축 단계는 4개 worker 병렬 wall time **4.864시간**, worker 시간 합 **18.995시간**입니다.
온라인 시간에는 포함하지 않았으며 실제 재사용 횟수에 따른 상각 검증은 남아 있습니다.
요청 Q=256에서 A의 실제 후보는 139~162개, B는 256개였습니다. 모든 비교군은 해당 조건에서
같은 실제 pose 수를 조회하지만 자세 간 LOS 재사용 때문에 ray 수까지 같지는 않습니다.

Q=128에서 큰 예산으로 늘렸을 때 diverse의 평균 coverage 증가는 **0.010%p**이고,
K 전체 평균 시간은 **133.53초 → 231.67초**였습니다. 144개 대응 조건 중 13개는 오히려 낮았습니다.
336개 원자료 후보군 쌍의 포함 관계는 통과했으므로 feasible 최적값이 낮아진 것이 아닙니다.
각 Q를 독립적으로 제한시간 solve하고 작은 Q의 해를 큰 Q의 초기해로 보존하지 않은 결과입니다.
원자료를 사후 최고값으로 덮어쓰지 않았습니다.

다음 우선순위는 (1) 모든 방법에 동일하게 작은 Q의 검증된 해를 큰 Q에서 보존하는 비교,
(2) 중첩 제약 전체 기준선의 메모리/초기해/상한 보강, (3) 새 geometry·장애물 seed에서 동일 시간의
quality-time 반복 비교입니다. 기준선 보강에 온라인 해를 사용한다면 독립 기준선과 별도 표시해야 합니다.
이번 결과만으로 library 수나 MAPPO 학습량을 먼저 늘려야 한다고 결론 내리지 않습니다.

### 분석 자료와 검증 범위

[상세 보고서](results/library_rsu_stress_20261123/analysis_20261124/REPORT.md),
[비교 그래프](results/library_rsu_stress_20261123/analysis_20261124/comparison.png),
[조건별 평균](results/library_rsu_stress_20261123/analysis_20261124/condition_means.csv),
[대응 차이](results/library_rsu_stress_20261123/analysis_20261124/paired_differences.csv),
[인증 원자료](results/library_rsu_stress_20261123/analysis_20261124/certificates.csv),
[검증 및 input/script hash](results/library_rsu_stress_20261123/analysis_20261124/validation_summary.json).

source 배치 240개와 대체 배치, 기준선 144개, 온라인 2,016개의 저장 mask/density로 coverage,
중첩률과 설치 제약을 재검산했습니다. 조회 mask의 전체 archive 일치, source/spec hash,
clear/mixed/dense의 동일 target 및 가림 단조성, bound/gap, CSV/JSON 일치를 검사했습니다.
기존 독립 direct-raycast 검증 기록 48개도 확인했습니다. **이번 분석에서는 새 raycasting,
학습 또는 solver를 실행하지 않았고 실험 원자료는 변경하지 않았습니다.**

분석 재현(기존 파생 결과도 덮어쓰지 않으므로 새 output 사용):

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  results/library_rsu_stress_20261123/analysis_20261124/analyze.py \
  --root results/library_rsu_stress_20261123 \
  --output results/library_rsu_stress_analysis_repeat
```

### 전체 후보 직접 재검증 완료 (2026-11-24)

**1단계의 구현·계산 일관성 검증을 통과했습니다.** `tests/validate_rsu_stress.py`로 당시
frozen source를 읽어 source 24개와 target/가림 24개, 총 48개 장면을 다시 생성했습니다.
선택된 배치만이 아니라 **전체 후보 mask 96,060개**를 새 direct raycasting으로 계산하여
저장 archive와 비트 단위로 일치함을 확인했습니다. 실제 제출 ray는 398,226,054개이며,
검증 전체는 약 181초였습니다. 이 시간은 성능 실험의 온라인 시간과 비교하지 않습니다.

| 검사 | 결과 |
|---|---|
| 전체 후보 pose/target/critical 집합 및 새 관측 mask | 48장면, 96,060개 mask 일치, 불일치 0 |
| Source 도로 상대 좌표 재생성 | 24/24 일치 |
| Source 검색·후보 전이 재실행 | 24/24 일치; 전이 중 물리 query/solve 금지 |
| 가림 단계 간 동일 후보 조회 계획 | 448개 대응 검사 통과 |
| Source 240개·기준선 144개·온라인 2,016개 배치 및 대체 해 | coverage/중첩/간격/pole/K와 저장 인증 일관성 통과 |
| 분석 재실행 | CSV 17개 byte-for-byte 동일 |
| 관련 회귀 테스트 | 105개 통과; 작은 MILP 전수조사 및 optical/전이/손상 검출 포함 |
| 기존 CLI 및 문법 | MILP/GREEDY/MAPPO dry-run 3개 및 py_compile 통과 |
| 원자료 및 frozen source | 검증 전후 hash 일치; 대조한 현재 source도 frozen hash와 일치 |

검증 보고서: [REPORT.md](results/library_rsu_stress_validation_20261124/REPORT.md),
[검증 범위·수치·hash](results/library_rsu_stress_validation_20261124/validation.json),
[테스트 명령/결과](results/library_rsu_stress_validation_20261124/tests.json).

**이 검증은 같은 Open3D/광학 모델의 새 계산입니다.** 독립적인 센서 물리 모델이나 실제 설치
정확성을 검증한 것은 아닙니다. 대규모 solver/학습을 재실행하지 않았고 원자료와 성능 수치는
변경하지 않았습니다. 회귀 테스트의 작은 solve/PPO 실행은 정식 성능 실험과 구분합니다.
큰 문제의 solver 상한은 저장 기록의 일관성을 검사했으며 독립적으로 다시 증명하지 않았습니다.
따라서 모든 library 해의 근사최적성, 동일 시간 우위, 새 실도로 일반화는 여전히 후속 검증 대상입니다.

재검증 명령(기존 실험 root 밖의 새 output 필요):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tests/validate_rsu_stress.py \
  --root results/library_rsu_stress_20261123 \
  --analysis results/library_rsu_stress_20261123/analysis_20261124 \
  --output results/library_rsu_stress_validation_repeat
```

## 이전 K=8 확장 구현 기록

**2026-11-23 구현 당시 기록: K=1~8·복수 source·후보 확대, 아래는 정식 실행 전 계획/검증 기록입니다.**
[확장 명세](LIBRARY_EXPANDED_PROTOCOL.md)와 [설정](configs/library_expanded_20261123.json)을 추가했습니다.
A 6개 + B 6개 source를 새로 구축하고, 가까운 source 1/3개 및 후보 예산 32/64/128개를 비교합니다.
기존 후보 수는 고정 10개가 아니라 7~20개였습니다. 새 요청 예산보다 실제 전이 후보가 적으면
임의로 채우지 않고 `actual_q`를 표시하며 모든 비교군이 같은 수를 실제 raycast합니다.
K=8은 최대 8대 예산이며, 적은 대수로 포화하면 실제 설치 수를 따로 기록합니다.

source와 전체 후보 기준선의 MILP는 **K별 최대 1,800초**입니다. 정확한 인증이면 일찍 종료하고,
그 외 조기 종료는 **300초 이상 + 120초 개선 없음 + 상대 gap 1% 이하**를 모두 요구합니다.
정체만으로 근사최적이라고 표시하지 않습니다. 각 후보군의 온라인 선택은 300초,
geometry portfolio의 비가림 surrogate 선택은 10초입니다. 이전 전수조사 경로는 그대로 유지했습니다.
각 K의 coverage·상한·gap 시간 곡선(`quality_time.csv/png`), 중첩률(`density>=2` / 전체 target),
실제 설치 수, 전처리·선별·raycast·최적화·solver 시간을 저장합니다. 중첩은 보고 지표이며 목적함수는
여전히 union coverage입니다. 후보군 내부 인증과 전체 유효 후보 대비 인증을 구분합니다.
새 기준선이 시간 제한이면 손실을 확정값 대신 상·하한으로 보고하며, 원래 기준선 표를 대체하지 않습니다.

실행 경로: `library_expanded_study.py` (`--execute` 없으면 계획만 출력),
짧은 기능 검증은 `--smoke --execute`입니다. 정식 설정의 12 source/16 target 조건 및 5,376행은
**계획 수량**이며 완료 결과가 아닙니다. 긴 계산의 상한과 시간 집계 범위는 확장 명세를 확인하세요.
검증: 신규 11개·기존 cut-MILP 4개·pipeline 7개, 문법 및 기존 CLI dry-run 3개 통과.
최종 축소 실행 [manifest](results/library_expanded_smoke_20261123_v3/manifest.json)는
source 2개/평가 8조건/1,152행 완료이며 저장 density·중첩·독립 raycast 검사를 통과했습니다.
[K별 시간 곡선 예시](results/library_expanded_smoke_20261123_v3/library/a_t_junction_w8/quality_time.png)와
[coverage·중첩·시간 그래프 예시](results/library_expanded_smoke_20261123_v3/comparison_A_s1_q32.png)는
짧은 제한시간의 **기능 검증 자료**이지 확장 library의 성능·속도 우위 근거가 아닙니다.

**2026-11-23: 설치 제한 × 가림 실험 224/224 완료 및 결과 분석.**
[설치 제한 실험 명세](LIBRARY_SITING_PROTOCOL.md)와
[고정 설정](configs/library_siting_large_20261123.json)을 추가했습니다.
기존 A/B source 및 56개 geometry를 유지하고 **설치 제한 유무 × 가림막 유무**로
224개 작업을 구성했습니다. K=1~3, library·geometry 3종·random 30 seeds의 비교는
22,848행, 전체 유효 후보 기준선은 별도 672행입니다. **13:55:06 KST 완료**, 배치 worker
시간은 1,036.05초입니다. 가림 비교 112쌍과 설치 제한 비교 112쌍의 검증을 통과했고
원본 library는 변경되지 않았습니다. 이번 분석에서는 학습·성능 실험·raycasting을 재실행하지 않았습니다.

| 방법 | 평균 coverage | 해당 유효 후보 최적값 대비 평균 손실 | 최적값 일치 조건 수 |
|---|---:|---:|---:|
| Library | 67.958% | 1.501%p | 310/672 |
| Geometry single | 65.304% | 4.155%p | 409/672 |
| Geometry union | 69.249% | 0.209%p | 543/672 |
| Geometry portfolio | 69.256% | 0.203%p | 573/672 |
| Random 30-seed 평균 | 67.416% | 2.043%p | seed 원자료 5,777/20,160 |
| 전체 유효 후보 기준선 | 69.459% | 0 | 672/672 |

56개 geometry × 4조건 × 3개 K를 동일 가중치로 집계한 기술통계입니다. Random은 먼저
각 조건의 seed 평균을 구했습니다. 최적값은 **22~72개 축소 이산 후보**에 대한 값이며
연속 공간·도시 전체 최적값이 아닙니다. 같은 조건의 물리 후보 조회 수만 동일하고, geometry
방법의 전체 후보 광학 계산·portfolio 전수조사 비용까지 같지는 않으므로 speedup을 주장하지 않습니다.

**해석:** Library는 single-score/random보다 전체 평균은 높지만, union/portfolio 대비
일관된 이점은 없습니다. Portfolio 대비 28승/305동률/339패입니다. K=3만 보면 기준선
92.342%, library 90.823%, portfolio 92.090%입니다. A의 평균 손실은 2.214%p,
B는 0.966%p이며 서로 다른 geometry·target 집합이므로 A/B 우열 표로 해석하지 않습니다.
B는 제한·가림 없는 96조건에서 전부 최적값과 일치했지만, 두 요인을 함께 넣으면 평균 손실
1.793%p, 최대 11.667%p입니다. B 전체에서는 library가 random 평균보다도 0.200%p 낮습니다.
112개 clear/occluded 쌍 모두 후보-target 관측 edge 감소가 있어 가림 입력은 실제로 작동했습니다.

**병목:** 조회한 후보군 내부는 이미 전수조사 최적입니다. 양의 손실은 library 후보군에
전체 최적값을 달성할 조합이 없다는 뜻이며 동일 후보 ID 안에서 LS/MAPPO를 더 돌려도
없어지지 않습니다. 기존 후보 밖으로 이동·보충하는 방법은 별도 평가가 필요합니다.
다음 후보는 고정 총 조회 예산의 library+geometry 보완과 geometry-only 비교이며,
이 실험을 본 뒤 정하는 규칙은 새 평가 지도에서 다시 검증해야 합니다. 새 방법의 효과는 아직 TBD입니다.

자료: [분석 보고서](results/library_siting_large_20261123/analysis_20261123_v2/REPORT.md),
[672조건 비교표](results/library_siting_large_20261123/analysis_20261123_v2/condition_summary.csv),
[22,848행 원자료 집계](results/library_siting_large_20261123/analysis_20261123_v2/raw_comparison.csv),
[요인별 대응 차이](results/library_siting_large_20261123/analysis_20261123_v2/paired_effects.csv),
[검증·집계·hash](results/library_siting_large_20261123/analysis_20261123_v2/summary.json).
`library_siting_report.py`는 원자료 hash, 저장 mask union, certificate 기록, CSV/JSON 일치를 검사합니다.
새로운 독립 물리 검산이나 최적화 전수조사를 다시 수행한 것은 아닙니다.
첫 집계의 JSON scalar 직렬화 오류를 수정한 뒤 새 `_v2` 보고서 폴더로 완료했습니다.
첫 파생 CSV는 보존했고 실험 원자료는 변경하지 않았습니다. 집계 테스트 4개, 기존 집계 회귀 24개,
pipeline 7개 및 문법 검사가 통과했습니다. 생성 결과는 ignored 경로이므로 clone에 포함되지 않습니다.

분석 재현(새 `--output` 사용):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_siting_report.py \
  --root results/library_siting_large_20261123 \
  --output results/library_siting_large_20261123/analysis_repeat
```

설치 제한은 도로 한 구간의 닫힌 XY 금지 영역이며 해당 위치의 모든 높이·자세 후보를
공통으로 제외합니다. target과 mesh는 바꾸지 않습니다. 전이 후 금지 후보를 제거하며
다른 위치로 몰래 재투영하거나 후보를 보충하지 않습니다. 전체 배치가 불가해도 그 배치의
남은 유효 후보는 조합 후보로 유지합니다. 제외 전후 후보 ID 대응과 library 후보 손실을 저장합니다.
동일 조건의 방법끼리는 조회 후보 수가 같지만, 설치 제한 전후에는 그 수가 달라질 수 있습니다.
실제 설치 기준이 정해지지 않았으므로 이 금지 구간은 **합성 기능·스트레스 시험 조건**입니다.
비용·간격·중복률·RL은 함께 추가하지 않았으며, 연속 도로 전체 최적성이나 도시 규모 실험이 아닙니다.

계획 확인은 다음과 같습니다. 실제 계산은 새 `--output`과 `--execute`가 필요합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_occlusion_study.py \
  --spec configs/library_siting_large_20261123.json
```

실행 전 검증 기록: 새 설치 제한 테스트 9개를 포함한 **88개 테스트 통과**, 문법 검사 및 기존
GREEDY/MILP/MAPPO dry-run 3개 통과입니다. 실제 source를 읽은 224개 작업 계획에서
후보 22~72개, 방법별 공통 조회 예산 7~20개, 최대 부분집합 62,269개로 전수조사 상한
100,000개 안에 있음을 확인했습니다. 이는 **축소 후보 영역**의 사전 검증이며 성능 결과가 아닙니다.
[검증 기록](results/library_siting_preflight_20261123/tests.json)과
[전체 계획](results/library_siting_preflight_20261123/plan.json)을 보존했습니다.
테스트에는 작은 임시 장면의 실제 raycast·전수조사와 4조건 frozen worker 실행이 포함됩니다.
사전 검증 당시에는 성능 분석을 하지 않았으며, 완료 후 분석은 위 별도 보고서에 정리했습니다.

**11월 22일 감사 당시 기록:** 아래는 확장 전 코드·계산 감사 시점의 상태입니다.
해당 문서 갱신에서는 새 성능 실험을 실행하지 않았습니다.
**11월 19일 완료 기록:** 비교 조건을 문서로 고정한 뒤 사용자 요청으로 기존 4개
개발 맵의 동일 조건 재현 실행을 완료했습니다. 새 조건·학습·대규모 MILP는 추가하지 않았습니다.
최근 대규모 완료 실험은 2026-11-09의 468-worker
mechanism study이고, 연구 검토는 2026-11-16의 ARS 평가입니다. 11월 18일 A/B 축소 source,
target 4개 및 동일 조회 예산 비교도 완료했습니다. 이는 작은 후보 영역의 전수조사이며
새 정식 학습·대규모 MILP·일반화 성능 실험이 아닙니다.

**11월 22일 확장 전 감사:** [LIBRARY_PRE_EXPANSION_AUDIT.md](LIBRARY_PRE_EXPANSION_AUDIT.md).
회귀 테스트 131개와 저장 mask에 대한 별도 정수 bitset 전수조사 420행이 통과했습니다.
감사에서 발견한 A 입력의 미지원 제약/장애물 무시와 제한 FOV의 추가 forward 조건은 후속 수정했습니다.
A source/target scenario의 미지원 필드를 거부하고, 직접/캐시/MAPPO/geometry-only 관측 판정을
공통 각도·거리 함수로 통일했습니다. 제한 FOV는 `angular_fov_range_v2`로 hash/라이브러리 버전을
구분하며 이전 제한 FOV library는 재구축해야 합니다. 기존 360도 library와 hash는 유지합니다.
수정 후 회귀 194개 통과, opt-in 전체 후보 parity 검사 1개 skip, 문법 검사 및 dry-run 7개 통과입니다.
기존 A/B 151개 후보를 새 scene에서 재계산해 저장 mask와 전부 일치했고 420행 전수조사도 일치했습니다.
검증 기록: [physical_recheck.json](results/library_audit_fixes_20261122/physical_recheck.json).
B의 현재 두 target은 모든 후보에서 실제 가림 손실이 0이므로 가림 대응 실험으로 해석하지 않습니다.
기존 결과 파일은 보존했습니다. 이번 수정은 입력 보호와 광학 판정 수정이며 제약 기능 추가나 새 성능 실험이 아닙니다.

**현재 결정: 먼저 고정 비교, 이후 필요한 부분만 확장.**
[LIBRARY_COMPARISON_PLAN.md](LIBRARY_COMPARISON_PLAN.md)에 A/B별 기존 source·대안 배치,
K=1~3, 물리 모델·후보 격자·선별 방법과 비교 기준을 기록했습니다. A와 B의 우열이 아니라
각 트랙에서 library가 강한 geometry-only 기준선보다 어떤 이점을 주는지 확인합니다.
완료한 4개 개발 맵 결과는 기준 자료로 보존하며 이번 재현을 새 조건·일반화 실험으로 세지 않습니다.
library 증설·가림막·비용·운영 제약·RL을 한꺼번에 추가하지 않고, 관측된 한계에 맞춰 한 요인씩
별도 설계합니다. 이후 가림 실험은 [별도 명세](LIBRARY_OCCLUSION_PROTOCOL.md),
설치 제한 추가는 위 11월 23일 명세로 구분합니다. 코드 준비와 실제 성능 실행 완료는 구분합니다.

**11월 19일 요청 재현 완료:** [실행 명세](results/library_query_controls_repeat_20261119/run_spec.json),
[완료 보고서](results/library_query_controls_repeat_20261119/report/REPORT.md),
[12조건 비교표](results/library_query_controls_repeat_20261119/report/condition_summary.csv).
4개 맵 × 34개 방법/seed = 136개 조합, K별 408행과 전체 기준선 12행을 별도 폴더에 저장했습니다.
실행 중 중간 로그·성능 분석 없이 종료 후 확인했습니다. 사전 테스트 131개(7+7+117), 기존 CLI
3개와 비교 4개 dry-run, 최종 집계 검증이 통과했습니다. 이전 11월 18일의 408행 원자료·12행
요약 및 네 지도 target/mesh 배열과 정확히 일치합니다. Library/Union/Portfolio의 전체 축소
후보 최적값 일치는 각각 10/12로 같으며, library의 일관된 우위라는 새 근거는 아닙니다.
재현 명령은 `bash results/library_query_controls_repeat_20261119/run.sh`이고 현재 완료 폴더에서는
재실행을 거부합니다. 반복하려면 새 결과 root에 실행 명세와 script를 보존해야 합니다.
이 실행은 source 재구축·새 맵 평가·시간 성능 통제 실험이 아닙니다. 기존 결과는 변경하지 않았습니다.

**현재 구현 기본 경로 (연구 우선순위와 구분):** source 자료 없이 geometry-only로 후보를 선별하고,
실제 raycasting과 공통 제약으로 배치를 선택합니다. 독립 CLI는 `geometry_placement.py`이며,
source 전이는 적은 조회 예산에서의 추가 효과를 확인하는 보조 실험으로 유지합니다.
아직 두 구간·고정 K=2의 파일럿이며 큰 도로망 전체 최적화로 확장한 것은 아닙니다.
CLI 기본값은 여전히 `adaptive`, 위치당 4개 조회, 반복 1회입니다.
`adaptive_batch`, `static_pair`, `no_feedback`, `random`, `full`은 명시적으로 선택합니다.

**저널 목표 연구 설계 (현재 TR-C 검토, 이전 T-ITS 계획 보존):** [TITS_RESEARCH_PLAN.md](TITS_RESEARCH_PLAN.md)의
최신 결정은 A/B library 구축·전이·조합 검증을 우선하며, library 학습은 후속 선택지로 보류합니다.
이전 source-free 계획과 Gate 1 결과는 보존합니다. pair-aware 정적 기준선과 피드백 제거 비교에서
**관측 피드백의 추가 성능 이점은 없었습니다.** partial-visibility bound는 미구현 보조 연구 후보로
남기며 library 구축의 선행 조건으로 두지 않습니다. 노벨티/게재 가능성은 미확정이고 검출은 후속입니다.

**통합 연구 방향 (2026-10-20):** 여러 배치 후보를 geometry·연결 조건에 맞게 전이하고,
전체 예산 아래 조합한 뒤 필요한 구간만 보정하는 방향으로 확장합니다.
첫 연결부 구현은 [CONNECTED_DEPLOYMENT_PROTOCOL.md](CONNECTED_DEPLOYMENT_PROTOCOL.md)의
두 구간·센서 2대·7개 꺾임 각도 파일럿입니다. 검출은 제외하며, 큰 도로망 최적화·원통형
반복 경계·곡선용 MAPPO·학습된 response 전이는 아직 구현한 기능이 아닙니다.
기존 410xxx library를 바꾼 것이 아니라 별도의 축소 직선 source를 계산해 비교합니다.

> 공통 엔진 리팩터링 1차 구현: Roadside/Robot application CLI와 가중 GREEDY/MILP를
> 추가했습니다. 구조, 재현 명령, 지원 범위는 [REFACTOR_PROGRESS.md](REFACTOR_PROGRESS.md)를
> 확인하세요. 로컬 UI는 `.venv/bin/python placement_ui.py --port 8766`으로 실행합니다.
> 복수 알고리즘/K 비교와 실행별 새 MAPPO 학습을 지원합니다.
> 지도/가중치 조건부 사전학습·전이 RL은 아직 구현되지 않았습니다.

**기초 배치 문제:** 제한된 수의 roadside LiDAR를 어디에, 어떤 높이와 방향으로 배치하면 도로의 관측 가능 영역을 최대화할 수 있는가?

이 프로젝트는 GREEDY, MILP, MAPPO를 공통 도로 geometry와 visibility/coverage 기준으로 비교합니다. 현재는 **정적 파라메트릭 도로의 기하학적 관측 범위 최적화**입니다. 동일 topology의 미사용 도로 폭에 대한 배치 전이·보간과 국소 재학습 실험은 완료했습니다. 실제 검출 정확도, 동적 차량 가림, 임의 실도로/미지 topology로의 일반화, 사전학습 정책 전이, 도시 전체 최소 센서 수 최적화는 검증한 기능이 아닙니다.

<a id="current-status"></a>

## 통합 진행 현황 (2026-11-19)

아래 통합 명세가 현재 상태입니다. 뒤의 날짜별 절은 당시 조건·결과·명령을 보존한 기록이며,
그 안의 "현재/다음/미완료"는 해당 실행 시점을 뜻합니다. 서로 다른 격자·목적함수·지도·시간 경계를
하나의 우열 표로 합치지 않습니다. `results/`는 ignored 경로이므로 clone만으로 산출물이 포함되지 않습니다.

### A/B Library의 대수별 저장 기준

현재 새 source 저장 계약은 기본 설치 가능 후보만 사용한 **무가중 union coverage, at-most-K**입니다.
비용·최소 간격·pole 용량 같은 추가 공동 제약은 아직 이 빌더에 적용하지 않습니다.

```text
f(S) = |union(i in S) A_i|
f*(K) = max f(S), S subset D_G, |S| <= K
L_K = 저장한 최고 feasible 배치의 관측 target 수
U_K = 같은 후보 영역에서의 유효 coverage 상한
L_K <= f*(K) <= U_K
```

- `budget_k`와 실제 `num_selected`를 분리하고 XYZ/yaw/pitch 및 도로 상대좌표를 저장합니다.
  K=6에서도 5대로 같은 coverage를 얻으면 실제 5대 배치를 보존할 수 있습니다.
- 이전 K의 해·GREEDY·제공된 feasible 배치·허용된 작은 전수조사 결과 중 최고를 보존합니다.
  K별 배치는 서로 달라도 되며 coverage는 감소하지 않습니다. 동률이면 적은 대수를 우선합니다.
- `L_K=U_K`이면 선언된 이산 후보에서 `discrete_optimal`, 아니면 `best_known`입니다.
  상대 gap은 `(U_K-L_K)/L_K`이며 coverage %p가 아닙니다. near-optimal 허용치는 아직 정하지 않았습니다.
- 대체 배치는 최고 해와의 손실을 함께 저장하며, 모두 최적·근사최적이라고 표시하지 않습니다.
  같은 mask라도 pose가 다르면 보존할 수 있지만 모든 branch/side 배분의 다양성을 보장하지 않습니다.
- 전체 후보의 mask union인 **도달 가능 coverage 상한**과 **모든 target 100%**를 구분합니다.
  상한은 joint 설치 제약이 없는 현재 후보 영역에만 유효하며, 실도로 전체의 상한이 아닙니다.
- 목표 Q를 달성한 m대 배치는 최소 대수의 상한 m입니다. K대 이하 coverage 상한이 Q보다 작으면
  최소 대수는 K+1 이상입니다. 두 대수 경계가 같을 때만 최소 대수로 인증합니다.
  단순한 K별 점수 정체나 미실행 K는 최소 대수의 증거가 아닙니다.

공통 구현은 `src/core/deployment_library.py`입니다. 첫 구현은 작은 문제의 전수조사 또는
GREEDY/제공 배치와 분석적 상한이며 **새 MILP 연결은 아직 없습니다**. 전수조사 상한을 넘으면
후보를 몰래 줄이지 않고 그 사실과 인증 수준을 기록합니다. 기존 MILP library 원자료는 보존합니다.

### A/B Source 저장 점검 (2026-11-18)

`library_topology.py`와 `library_components.py`를 추가하고 축소 source 두 개에서 K=1,2,3을
실행했습니다. **이 source 저장 실행 자체는 새 지도 전이·검색·조합·학습 실험이 아닙니다.** A는 기존 파라메트릭 T자,
B는 단일 곡선 모듈입니다. 이 형식은 기존 `prototype_transfer.py`의 `kK/` loader와 바로 호환되지 않습니다.

| Source 저장 점검 | 후보 / target | K=1 coverage | K=2 coverage | K=3 coverage |
|---|---:|---:|---:|---:|
| A: T자 7m, 단일 높이·자세 및 10m 대표 bin | 28 / 3,544 | 53.837% | 75.423% | 98.900% |
| B: 길이 80m·폭 8m·회전 30도·원호 24m | 24 / 160 | 76.250% | 99.375% | 100.000% |

두 source 모두 각각의 **작은 선언 후보 영역**에서 전수조사를 완료해 6개 K 행의 gap은 0입니다.
연속 공간·기존 전체 격자·실도로의 최적값이 아닙니다. A/B는 서로 다른 문제이므로 위 표로 우열을 비교하지 않습니다.
최고 배치와 K별 최대 3개 대안(최고 배치 포함)의 좌표·mask union을 별도 raycast provider로 확인했습니다.
모든 후보를 다시 독립 구현으로 raycasting한 검증은 아닙니다.

B에서는 해당 영역/target의 100% 달성 최소 대수가 3으로 인증됩니다. A에서는 100% 달성에
**4대 이상 필요**하다는 하한만 있고, 이번에 저장한 K<=3에는 100% 배치가 없어 최소 대수는 미확정입니다.
현재 파일: [A frontier](results/library_ab_smoke_20261118/a_v2/frontier.json),
[B frontier](results/library_ab_smoke_20261118/b/frontier.json).
A의 이전 `a/` 실행도 보존했으며 `a_v2/`는 실행 표시·좌표 형식 metadata를 보완한 같은 조건의 실행입니다.

아래 명령은 이미 사용한 output이므로 재실행에는 새 이름이 필요합니다. `--execute`를 빼면
파일·raycast·최적화 없이 계획만 확인합니다. source code/input snapshot과 hash, 전체 후보 배열,
상대좌표, phase별 시간과 검산 기록을 각 실행에 저장합니다. 생성 결과는 git ignored 경로입니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_topology.py \
  --config configs/library_topology_smoke.yaml --scenario-id 1 --k-values 1 2 3 \
  --output results/library_ab_smoke_20261118/a_v2 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_components.py \
  --length-m 80 --road-width-m 8 --turn-deg 30 --bend-length-m 24 --mesh-step-m 2 \
  --station-step-m 20 --candidate-heights-m 3 --relative-yaws-deg -30 0 30 --pitches-deg -10 \
  --target-step-m 2 --target-heights-m 1.5 --max-k 3 --alternatives-per-k 3 \
  --enumeration-limit 100000 --output results/library_ab_smoke_20261118/b --execute
```

새 테스트는 공통 계약 7개, 독립 전수조사 audit 19개, A 빌더 5개, B 빌더 10개입니다.
기존 pipeline 7개와 함께 **48개 검사 통과**, Python 문법 검사와 기존 CLI 3개 dry-run도 통과했습니다.
검사 로그는 [validation_tests.log](results/library_ab_smoke_20261118/validation_tests.log)입니다.
구축 성능이나 재사용 이점은 별도로 평가하며, 이번 단일 실행 시간을 speedup으로 보고하지 않습니다.

### 새 형식 A 전이 / B 조합 검증 (2026-11-18)

`library_transfer_study.py`로 **새 target 4개 × K=1,2,3 = 12개 조건**을 완료했습니다.
원본 source는 바로 위 `a_v2/`, `b/`로 고정했고 변경하지 않았습니다. 이번은 작은 합성 개발
문제이며 실제 도로망·독립 최종 시험셋·학습 실험이 아닙니다. source 자동 검색도 아직 없습니다.

- **A:** T자 7m source를 폭 9m/교차점 (45,55), 폭 12m/교차점 (55,45)로 각각 전이합니다.
  branch/side를 유지하고 설치 구간 비율 `u`를 새 구간에서 복원한 뒤, 선언된 target 격자로
  기하 투영합니다. source의 K별 최고 배치와 대안을 각각 전이하며 강제 좌표 평균은 하지 않습니다.
- **B:** 길이 80m source를 160m target의 두 station 구간에 각각 적용합니다. target 폭은 10m,
  회전은 45도 또는 60도, 원호 길이는 32m입니다. source station 비율·side·상대 자세를 복원하고
  geometry로 먼저 만든 전체 후보 격자로 투영합니다. 두 구간은 동일 scene에서 서로의 영역을
  관측할 수 있으며 경계 wrap은 없습니다. **임의 T/십자 도로망 분해나 서로 다른 모듈 유형 조합은 아닙니다.**
- 두 트랙 모두 성공한 전이 배치들의 **개별 pose 합집합**을 후보 pool로 사용합니다. source의
  배치 한 묶음만 고르는 방식이 아니라 여러 대안/K의 pose를 재조합합니다. B도 구간별 quota 없이
  전체 at-most-K로 선택합니다. source의 관측 mask·최적성 인증을 target에 복사하지 않습니다.
- 먼저 pool만 새 target에서 raycasting하고 배치를 확정합니다. 그 뒤 별도 provider로 전체 후보를
  평가해 전수조사/GREEDY 기준선을 계산합니다. 온라인 결과나 source 해를 전체 기준선의 초기해로 넣지 않습니다.
  전수조사를 생략하는 fallback에서도 pool GREEDY보다 좋은 전이 원본 배치는 보존합니다.

#### 전체 결과

단위 coverage %. 아래 전체 기준값은 **선언된 축소 이산 격자에서만** 전수조사 최적값입니다.
높이·자세·설치 sampling이 제한되어 있으므로 기존 full-grid MILP나 연속 공간의 최적값과 다릅니다.

| target | K | 전이 pool 공동 선택 | 전체 후보 최적값 | 전체 후보 GREEDY |
|---|---:|---:|---:|---:|
| A 폭 9m | 1 | 50.598 | 50.598 | 50.598 |
| A 폭 9m | 2 | 75.742 | 75.742 | 71.744 |
| A 폭 9m | 3 | 98.908 | 98.908 | 91.797 |
| A 폭 12m | 1 | 51.866 | 54.721 | 54.721 |
| A 폭 12m | 2 | 78.764 | 78.764 | 78.034 |
| A 폭 12m | 3 | 99.010 | 99.108 | 94.711 |
| B 45도 | 1 | 38.750 | 38.750 | 38.750 |
| B 45도 | 2 | 74.0625 | 74.0625 | 74.0625 |
| B 45도 | 3 | 98.750 | 98.750 | 98.750 |
| B 60도 | 1 | 40.3125 | 40.3125 | 40.3125 |
| B 60도 | 2 | 74.375 | 74.375 | 74.375 |
| B 60도 | 3 | 98.750 | 98.750 | 98.750 |

**10/12 조건에서 전체 후보 최적값과 일치**했습니다. 손실은 A 폭 12m의 K=1에서 176 targets
(2.855%p), K=3에서 6 targets (0.097%p)입니다. 이는 저장된 대안을 전이한 pool이 좋은 target
후보를 놓칠 수 있다는 실제 사례입니다. source가 최적이어도 target 최적성이 자동 보존되지는 않습니다.
pool 내 인증과 전체 후보 인증은 report에서 별도로 저장합니다.

| target | 전체 후보 → pool | target 수 | pool ray 수 / 전체 후보 ray 수 |
|---|---:|---:|---:|
| A 폭 9m | 30 → 11 | 4,852 | 23,232 / 55,507 |
| A 폭 12m | 25 → 8 | 6,164 | 21,781 / 60,122 |
| B 45도 | 48 → 20 | 320 | 2,115 / 5,050 |
| B 60도 | 48 → 20 | 320 | 2,131 / 5,092 |

후보 pose 평가 수는 58.3~68.0% 감소했지만 이는 **한 지도에서 K들을 공유하는 후보 평가 수**입니다.
K별 독립 cold-start 시간이나 speedup이 아닙니다. source 로드·전이, 공통 mesh, pool raycast/선택,
기준선 raycast/선택, 검산을 분리 계측했습니다. 같은 프로세스·공유 서버의 개발 실행이고 테스트
작업과도 시간대가 겹쳤으므로 정식 runtime 비교로 쓰지 않습니다. source 구축 비용도 별도입니다.

네 target 모두 이번 전체 후보 K<=3에서 100% 배치는 없었습니다. 100%를 위해 **최소 4대 이상**
필요하다는 하한만 확인했으며, 4대가 충분하거나 대형 도로망의 최소 대수를 구했다는 뜻은 아닙니다.
source B의 100% 최소 3대 인증을 target에 그대로 적용할 수 없다는 점도 확인됩니다.

#### 재현과 검증

출력은 [A 폭 9m](results/library_ab_transfer_20261118/a_w9/comparison.csv),
[A 폭 12m](results/library_ab_transfer_20261118/a_w12/comparison.csv),
[B 45도](results/library_ab_transfer_20261118/b_turn45/comparison.csv),
[B 60도](results/library_ab_transfer_20261118/b_turn60/comparison.csv)입니다.
각 폴더에 `report.json`, 실제 `target.npz`/`mesh.npz`, manifest/hash, 원본 library 사본,
실행 코드 snapshot 및 worker log를 보관합니다. 결과 폴더는 git ignored입니다.

아래는 실제 실행한 CLI 인자이며 재실행에는 새 output 이름을 사용합니다. `--execute`를 빼면
source 일관성·기하 전이·후보 포함만 검사하고 파일/target raycast/최적화를 생성하지 않습니다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
.venv/bin/python -B library_transfer_study.py --track A \
  --source results/library_ab_smoke_20261118/a_v2 --target-json configs/library_target_a_w9.json \
  --k-values 1 2 3 --output results/library_ab_transfer_20261118/a_w9 --execute
.venv/bin/python -B library_transfer_study.py --track A \
  --source results/library_ab_smoke_20261118/a_v2 --target-json configs/library_target_a_w12.json \
  --k-values 1 2 3 --output results/library_ab_transfer_20261118/a_w12 --execute
.venv/bin/python -B library_transfer_study.py --track B \
  --source results/library_ab_smoke_20261118/b --target-json configs/library_target_b_turn45.json \
  --k-values 1 2 3 --output results/library_ab_transfer_20261118/b_turn45 --execute
.venv/bin/python -B library_transfer_study.py --track B \
  --source results/library_ab_smoke_20261118/b --target-json configs/library_target_b_turn60.json \
  --k-values 1 2 3 --output results/library_ab_transfer_20261118/b_turn60 --execute
```

새 adapter 20개·공통 평가/runner 10개와 이전 48개를 합쳐 **78 tests 통과**했습니다.
[전체 검사 로그](results/library_ab_transfer_20261118/validation_tests_final.log).
최종 산출물의 원본/source/input/artifact hash, mask union, pose/실제 대수 및 CSV 일치도 확인했습니다.
192개 저장 배치 record는 최고 배치가 대안에 반복되는 항목을 포함하므로 독립 실험 192회가 아닙니다.
전이 record는 총 72개(중복 포함) 모두 생성됐고, source 원자료는 변경되지 않았습니다.
검산은 새 provider/scene의 실제 raycast이며 독립 물리 구현과의 비교는 아닙니다.

이 전이 검증에 이어 아래 동일 조회 예산 비교를 완료했습니다. 더 넓은 source geometry/배치 다양성,
사용하지 않은 target family, 반복 cold-start 시간 비교는 남아 있습니다. 추가 운영 제약·LS/RL
재배치와 library 학습은 아직 후속이며, library 우위나 논문 노벨티를 확정하지 않습니다.

### A/B Library 이점 검사: 같은 후보 조회 예산 (2026-11-18)

**결론: 후보를 줄여 좋은 배치를 얻었지만, library 고유의 일관된 이점은 아직 없습니다.**
같은 네 개발 지도에서 library와 geometry-only 세 방법, 무작위 30 seeds를 비교했습니다.
지도별 34개 방법·seed 조합, 총 **136개 조합 / K별 408행**을 완료했습니다. 전체 후보 기준값은
별도 12행입니다. source와 격자는 바꾸지 않았고 새 학습·검출·운영 제약은 추가하지 않았습니다.

사전 규칙은 [LIBRARY_QUERY_CONTROLS.md](LIBRARY_QUERY_CONTROLS.md)이며 실행마다 고정 사본을
보관합니다. 모든 방법은 지도별 같은 q개 pose를 실제 raycasting하고 그 pool에서 동일한
at-most-K 전수조사를 수행합니다. K=1,2,3은 같은 pool을 공유합니다.

| 방법 | 실제 raycast 전 후보 선별 정보와 규칙 |
|---|---|
| `library` | 기존 source의 K별 배치·대안 pose를 기하 전이한 pool 그대로 |
| `geometry_single` | 장애물 가림을 제외한 range/FOV 단일 관측 수 상위 q개 |
| `geometry_union` | 같은 비가림 예측의 추가 union을 순차 최대화, q개까지 채움 |
| `geometry_portfolio` | 같은 비가림 예측의 K별 최적/대안 배치를 열거하여 후보를 모음 |
| `random` | 전체 후보에서 중복 없이 q개, 고정 seed=0..29를 모두 보고 |

geometry/random 선별 함수에는 source 정보·target 실제 관측 mask·기준선 해를 주지 않습니다.
geometry 방법은 **전체 후보의 광학 예측**을 계산하며 portfolio는 예측 부분집합도 열거합니다.
따라서 raycast가 없다고 계산량이 0인 것은 아닙니다. 모든 query plan을 먼저 확정하고, 모든
방법의 배치를 확정한 뒤에만 전체 후보 기준선을 별도 계산합니다. 실행 중 로그는 읽지 않았습니다.

```text
q = |transferred_pool|                    # A9/A12/B45/B60: 11/8/20/20
L_method(K) = max f(S), S subset queried_pool_method, |S| <= K
full_shortfall_count = f*_full(K) - L_method(K)
library_minus_control_pp = 100 * (L_library(K) - L_control(K)) / N
```

#### 전체 비교 결과

단위 %. Random은 해당 지도/K에서 30개 후보 추출의 평균 ± 표본 SD(`ddof=1`)입니다.
이는 지도 30개의 반복이 아니며 통계적 일반화 검정도 아닙니다. 반올림 전 target 개수로 우열을 판정합니다.
전체 최적값은 여전히 **작은 선언 후보 영역**에 한정됩니다.

| target | K | Library | Single | Union | Portfolio | Random 평균 ± SD | 전체 최적값 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A 폭 9m | 1 | 50.598 | 50.598 | 50.598 | 46.311 | 50.104 ± 0.839 | 50.598 |
| A 폭 9m | 2 | 75.742 | 75.742 | 75.742 | 75.742 | 73.867 ± 1.541 | 75.742 |
| A 폭 9m | 3 | 98.908 | 97.176 | 98.908 | 98.908 | 96.780 ± 3.672 | 98.908 |
| A 폭 12m | 1 | 51.866 | 54.721 | 54.721 | 54.721 | 52.344 ± 2.727 | 54.721 |
| A 폭 12m | 2 | 78.764 | 77.677 | 78.634 | 78.634 | 76.983 ± 2.251 | 78.764 |
| A 폭 12m | 3 | 99.010 | 97.307 | 98.378 | 99.108 | 94.552 ± 5.146 | 99.108 |
| B 45도 | 1 | 38.750 | 38.750 | 38.750 | 38.750 | 38.406 ± 0.521 | 38.750 |
| B 45도 | 2 | 74.0625 | 74.0625 | 74.0625 | 74.0625 | 73.521 ± 0.449 | 74.0625 |
| B 45도 | 3 | 98.750 | 98.750 | 98.750 | 98.750 | 97.917 ± 0.708 | 98.750 |
| B 60도 | 1 | 40.3125 | 40.3125 | 40.3125 | 40.3125 | 39.938 ± 0.632 | 40.3125 |
| B 60도 | 2 | 74.375 | 74.375 | 74.375 | 74.375 | 73.917 ± 0.424 | 74.375 |
| B 60도 | 3 | 98.750 | 98.750 | 98.750 | 98.750 | 98.042 ± 0.715 | 98.750 |

| 비교군 | Library 승 / 동률 / 패 (12조건) | 평균 Library - 비교군 | 비교군의 전체 최적값 일치 |
|---|---:|---:|---:|
| Single | 3 / 8 / 1 | +0.139%p | 9/12 |
| Union | 2 / 9 / 1 | -0.174%p | 10/12 |
| Portfolio | 2 / 8 / 2 | +0.122%p | 10/12 |
| Random 평균 | 11 / 0 / 1 | +1.126%p | seed별 분포로 별도 보고 |

Library 자체의 전체 최적값 일치는 **10/12**입니다. 평균은 지도/K 동일 가중치의 기술통계이며
독립 지도 수나 통계적 유의성을 뜻하지 않습니다. geometry 세 방법 중 사후 최고값만 고른 비교도 아닙니다.

- A 폭 12m/K2에서는 library가 Union/Portfolio보다 8 targets(0.130%p) 높습니다.
  그러나 같은 지도의 K1에서는 세 geometry 방법보다 176 targets(2.855%p) 낮습니다.
- B의 여섯 조건은 네 결정적 방법이 모두 같은 전체 최적값입니다. 이 조건에서는
  **source library가 없어도 같은 조회 수로 같은 품질**을 얻었습니다.
- Random보다 좋다는 결과만으로 library 필요성을 입증하지 않습니다. Union의 평균이 더 높고,
  Portfolio와도 승패가 갈리므로 일관된 source 재사용 우위라는 주장은 보류합니다.

#### 계산량과 검증 범위

| target | 전체 후보 M / 공통 q | Library rays | Single rays | Union rays | Portfolio rays | geometry 방법당 광학 point tests |
|---|---:|---:|---:|---:|---:|---:|
| A 폭 9m | 30 / 11 | 23,232 | 27,617 | 25,882 | 22,297 | 145,560 |
| A 폭 12m | 25 / 8 | 21,781 | 26,750 | 25,515 | 23,858 | 154,100 |
| B 45도 | 48 / 20 | 2,115 | 2,352 | 2,352 | 2,200 | 15,360 |
| B 60도 | 48 / 20 | 2,131 | 2,394 | 2,394 | 2,314 | 15,360 |

q는 pose 조회 수이고 rays는 광학 필터를 통과해 실제 제출한 ray 수입니다. **동일 q는 동일 ray 수·
연산량·시간이 아닙니다.** 각 방법은 별도 provider를 쓰지만 mesh/process를 공유합니다. 선별,
raycast, 선택, source 로드·전이는 별도 계측하고 source 구축 비용은 추가입니다. 반복 cold-start,
동일 품질 도달 시간 또는 상각 실험이 아니므로 speedup을 계산하지 않습니다.

각 실행은 저장 배치를 새 raycast provider로 검산했습니다. 이후 별도 integer-bitset 전수조사로
**408개 방법/K + 12개 전체 기준값 = 420행**의 최적값과 동률 선택 ID를 재확인했습니다.
이는 저장 mask의 독립 조합 검산이지 다른 물리 구현에 의한 검산은 아닙니다.
source/input/artifact hash 검사, 원본 library 불변 확인을 통과했습니다. 이전 실행의
library/전체 기준값, 선택 ID, pose/target/mask 배열도 네 지도 모두 그대로 재현했습니다.

자료: [12조건 비교 CSV](results/library_query_controls_20261118/report/condition_summary.csv),
[408행 원자료 CSV](results/library_query_controls_20261118/report/raw_comparison.csv),
[집계/재현 검증](results/library_query_controls_20261118/report/summary.json),
[독립 전수조사 audit](results/library_query_controls_20261118/independent_enumeration_audit.json).
각 지도 `report.json`에는 seed별 query ID·ray 수·시간·선택 결과가 있으며, `inputs/protocol.md`와
source snapshot을 보존합니다. 결과는 ignored 경로라 clone만으로 포함되지 않습니다.

#### 재현 명령과 다음 단계

아래 Bash 명령은 실제 사용한 인자입니다. output이 이미 있으므로 반복 실행은 새 `ROOT`로 지정합니다.
각 호출에서 `--execute`를 빼면 파일 생성·후보 순위 계산·raycast·전수조사 없이 계획만 확인합니다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
ROOT=results/library_query_controls_20261118
SEEDS=({0..29})
for CASE in a_w9 a_w12 b_turn45 b_turn60; do
  case "$CASE" in
    a_*) TRACK=A; SOURCE=results/library_ab_smoke_20261118/a_v2 ;;
    b_*) TRACK=B; SOURCE=results/library_ab_smoke_20261118/b ;;
  esac
  .venv/bin/python -B library_transfer_study.py --track "$TRACK" --source "$SOURCE" \
    --target-json "configs/library_target_${CASE}.json" --k-values 1 2 3 \
    --control-seeds "${SEEDS[@]}" --output "$ROOT/$CASE" --execute || break
done
.venv/bin/python -B library_query_report.py \
  --roots "$ROOT/a_w9" "$ROOT/a_w12" "$ROOT/b_turn45" "$ROOT/b_turn60" \
  --output "$ROOT/report" --previous-root results/library_ab_transfer_20261118
.venv/bin/python -B -m unittest discover -s tests -p 'test_library*.py'
```

기존 pipeline 포함 실행 전 101개 테스트와 기존 CLI 3개/새 비교 4개 dry-run을 통과했습니다.
추가 집계 검사는 누락 seed·불일치 mask/좌표/CSV·예산·인증·이전 결과 재현 오류를 거부합니다.
집계 테스트 24개를 포함한 **최종 125개 테스트 통과(실패/오류/skip 0)**:
[검사 로그](results/library_query_controls_20261118/validation_tests_final.log).
README 비교표의 84개 수치도 CSV의 반올림 값과 대조했습니다.
11월 19일에는 [고정 비교 및 확장 규칙](LIBRARY_COMPARISON_PLAN.md)을 기록했습니다.
source 수/대안 다양성과 조회 예산의 영향을 분리하는 ablation, 미사용 geometry family 평가는
후속 후보입니다. 새 입력·조회 예산·선택 규칙을 먼저 확정하며 기존 결과의 프로토콜은 바꾸지 않습니다.
특히 library를 늘리면서 q도 자동 증가시키면 library 정보의 효과와 추가 조회 효과를 분리할 수 없습니다.
유리한 지도만 고르거나 이 결과를 보고 바로 RL 우위로 연결하지 않습니다.

| 읽을 내용 | 자료 |
|---|---|
| 계산 방식과 구현된 제약 | [공통 계산 명세](#current-math) |
| 실험별 완료 범위와 핵심 결과 | [전체 진행 이력](#progress-history), 뒤의 날짜별 재현 명령 |
| A/B 동일 조회 예산 비교 | [비교표](results/library_query_controls_20261118/report/condition_summary.csv), [고정 프로토콜](LIBRARY_QUERY_CONTROLS.md) |
| 이전 source-free 후보 선별 비교 | [Gate 1 결과](#mechanism-results), [고정 프로토콜](GEOMETRY_MECHANISM_STUDY.md) |
| 연구 수준과 남은 검증 | [ARS 평가와 다음 단계](#research-review) |
| 제안 상태인 bound·도로망·인프라 수식 | [미구현 연구 설계](#proposed-model) |
| 실행·검증·파일 책임 | [재현 및 코드 안내](#current-reproduction) |
| 기존 폭 전이/보간, MAPPO, MILP 상세 | 뒤의 날짜별 절 및 번호 4~10절. 기본값과 실행 snapshot을 구분 |

| 항목 | 구현 / 실행 상태 | 현재 근거가 허용하는 결론 |
|---|---|---|
| 공통 geometry·range/FOV·mesh LOS·coverage | 구현, 소형 테스트 및 저장 배치 검산 | 정적 모델의 유한 target 관측 여부. 실제 검출률 아님 |
| A/B 공통 K별 source 저장·전이 | 축소 source 2개 및 새 target 4개·K1~3 저장/검산 | target 10/12 조건에서 전체 축소 후보 최적값 일치. 속도 우위·일반화·자동 source 검색은 미검증 |
| A/B 동일 조회 예산 비교 | 136 방법·seed 조합, 408 K행 및 별도 전체 기준값 12행 완료 | Random 평균보다 11/12 높으나 강한 geometry-only 대비 일관된 library 우위 없음 |
| source pose의 도로 상대 전이·보간 | 구현, library·폭/교차점 변형 실험 완료 | 배치 재사용 가능; 대응 불가와 초기값 민감도 존재 |
| 동일 초기값 LS/MAPPO | 무가중 및 제약부 경로 실행 | 실행한 조건에서 MAPPO 추가 이점 미입증; 사전학습 정책 전이 아님 |
| 비용·설치·ROI·중복·N-1·hub 검사 | opt-in 공통 evaluator 구현 | 예시 조건 검증; 모든 조건을 최신 실험에서 켠 것은 아님 |
| 곡선 연결과 후보 공동 선택 | 구현, 두 구간·K=2 실행 | 구간 간 관측 중복을 같은 scene에서 평가; 도시 도로망 해법 아님 |
| source response / 가림 전이 | 구현, ablation 완료 | 일부 저예산 개선과 negative transfer; 일관된 source 우위 없음 |
| source-free 선택적 visibility | 구현, 180/108/264/468-worker 연구 각각 완료 | 조회 절감 확인, 같은 품질의 full 대비 시간 우위 미확인 |
| 쌍 상호작용과 관측 피드백 분리 | Gate 1 완료 | 피드백 제거군과 최종 배치·J가 36/36 동일 |
| partial-incidence bound 및 query rule | **Gate 2 제안, 미구현** | soundness·tightness·계산비용 모두 TBD |
| 다중 교차로·가변 전역 K·실제 인프라 graph | **미구현** | 큰 도로망 최적화·현장 적용·최소 센서 수 보장 없음 |
| 검출·추적·통신 QoS·실측 검증 | **범위 밖 / 후속** | 기하 coverage를 recall·tracking continuity·안전성으로 부르지 않음 |

<a id="current-math"></a>

## 공통 계산 명세와 수식

### 1. 배치, 관측, 목적값의 단위

후보 집합을 $\mathcal C$, 선택한 배치를 $P\subseteq\mathcal C$,
pose를 $p_i=(x_i,y_i,z_i,\mathrm{yaw}_i,\mathrm{pitch}_i)$,
유한 target 집합을 $T=\{q_j\}_{j=1}^N$로 둡니다. roll은 없습니다.
현재 후보와 target은 합성 geometry에서 생성하며 실제 OSM polyline을 복원한 결과가 아닙니다.

$$
a_{ij}=\mathbf 1[\mathrm{range/FOV}(p_i,q_j)]\,
        \mathbf 1[\mathrm{LOS}_{\mathrm{mesh}}(p_i,q_j)],\qquad
d_j(P)=\sum_{i\in P}a_{ij}.
$$

$$
F(P)=\sum_j\mathbf 1[d_j(P)\ge1],\qquad
C(P)=100\,F(P)/N,\qquad
O(P)=\frac1N\sum_j\mathbf 1[d_j(P)\ge2].
$$

`coverage_pct`는 $C$의 %, `overlap_ratio`는 $O$의 0~1 비율입니다.
**중복 패널티는 이중 관측 target 비율이 아니라 초과 관측 횟수**
$E(P)=\sum_j\max(d_j(P)-1,0)$입니다. K=2에서는 중복 target 수와 같지만 K>2에서는 다릅니다.
미관측 target 수는 $N-F(P)$입니다. 이 지표는 표본 비율이며 연속 면적·검출 정확도가 아닙니다.

[공통 목적함수](src/core/evaluation.py)는 자동 정규화하지 않은 count 기반 값입니다.
중요도 $\eta_j$는 기본 1이고 겹친 중요 영역의 factor는 최댓값을 적용합니다.

$$
J(P)=w_c F(P)+w_I\sum_j\eta_j\mathbf1[d_j(P)\ge1]
     -w_o E(P)-w_n|P|-w_b\sum_{i\in P}b(p_i).
$$

$$
b(p_i)=b_0+b_z z_i+\sum_r b_r\mathbf1[(x_i,y_i)\in R_r].
$$

비용은 절대 설치 높이와 포함 영역 surcharge를 센서별로 더한 **모형 단위**입니다.
실제 원화 견적·공용 지주/굴착비 공유 모델이 아닙니다. 가중치는 비음수이며
`robustness_weight`의 양수 설정은 아직 지원하지 않습니다. N-1은 아래 hard constraint입니다.
고정 K의 보정에서는 $w_n|P|$가 상수이므로 센서 수를 줄이는 효과가 없습니다.
최신 corridor의 실제 설정은 $w_c=1,w_o=0.05,w_b=0.2,w_I=w_n=0$,
$b(p_i)=10+z_i$입니다. **J 손실, coverage 차이(%p), solver 상대 gap(%)은 서로 다른 양**입니다.

### 2. 공통 물리 판정과 선택적 조회

[MeshVisibility](src/core/visibility.py)는 target 방향의 직선 ray를 mesh에 발사합니다.
센서-target 거리 $r_{ij}$, 첫 충돌 거리 $t_{ij}$에 대해 현재 LOS 판정은
$t_{ij}$가 비유한값이거나 $t_{ij}\ge r_{ij}-10^{-3}\mathrm m$인 경우입니다.
range/FOV 각도 회전식은 뒤의 번호 6절에 보존했습니다. 최신 profile은
`VELODYNE64_R30`: 최대 거리 30m, 수평 FOV 360도, 수직 FOV 26.9도입니다.
실제 64개 beam의 시간별 발사·point density·반사율은 모델링하지 않습니다.

가림 없는 예측을 $g_{ij}=\mathbf1[\mathrm{range/FOV}(p_i,q_j)]$라 하면,
동일 geometry·FOV 구현에서 $a_{ij}\le g_{ij}$입니다. $g_{ij}=1$만으로 관측을 인증할 수 없습니다.
문서/코드의 `measured`는 **정적 mesh에서 계산한 mask**이며 실제 LiDAR 장비 측정이 아닙니다.
한 pose 조회는 여러 target ray를 포함하고, FOV/range에서 제외된 target에는 ray를 생략합니다.
따라서 pose 조회 수, 제출 ray 수, optical point tests, pair tests, CPU 시간은 같지 않습니다.

최신 제한 방법은 선택한 pose만 raycast하고 해당 worker 안에서 memoization합니다.
최종 배치는 별도의 scene에서 direct raycast로 확인합니다. full reference의 mask는
제한 방법에 주지 않습니다. geometry와 target 좌표 자체는 알려진 입력이지 온라인 SLAM 결과가 아닙니다.

### 3. 구현된 hard constraints와 실제 활성 설정

[CoverageRequirements](src/core/coverage_requirements.py)와
[CorridorRequirements](src/roadside/corridor_requirements.py)가 공통 판정을 담당합니다.
제약 위반과 큰 목적값을 상쇄하지 않으며, feasible 배치 안에서 J를 최대화합니다.

| 조건 | 수학적 / 구현상 의미 | 최신 11월 9일 실험 |
|---|---|---|
| 수·설치 가능성 | $|P|\le K$, 중복 pose 금지, 공통 설치 격자·높이·자세 | pair selector는 **정확히 2대**; 일반 at-most-K solver와 구분 |
| 설치비용 | $\sum_i b(p_i)\le B$ | $B=28$ |
| 이격·지주 용량 | 센서 쌍 XY 거리 ≥ 최소값, 같은 XY의 설치 수 상한 | 3m 이상, 같은 XY 최대 1대 |
| 허용 높이·영역 | height window, 허용/금지 직사각형; corridor는 station/side 금지 | 후보 높이 3/5m; combined 금지 구간 추가 |
| 전체 / ROI coverage | $C_R=100\sum_{j\in R}\mathbf1[d_j\ge1]/|R|\ge\tau_R$ | 전체 ≥60%; combined 연결부 ≥95% |
| 전체 / ROI 중복 | $\rho_R^-\le O_R=\sum_{j\in R}\mathbf1[d_j\ge2]/|R|\le\rho_R^+$ | 전체 ≤0.8; combined 연결부 ≥0.5 |
| 단일 센서 고장 | $C_{-1}=\min_{i\in P}100\sum_j\mathbf1[d_j-a_{ij}\ge1]/N$ | 전체 ≥25%; 센서별 mask 필요 |
| 서비스 hub | 수평 도달거리 안의 hub에 각 센서를 할당하는 용량 제한 이분 매칭 | 공통 v2에는 구현, **최신 corridor에는 비활성** |

ROI의 분모는 해당 영역의 모든 높이 target 수입니다. N-1은 모든 선택 센서 중 하나를 제거한
최악값이며 독립 고장 확률·공유 전원 고장 모델이 아닙니다. hub 매칭은 포트 수와 수평 거리만
검사하므로 케이블 경로·전력량·대역폭·지연시간을 보장하지 않습니다.

최신 `combined`는 station [0,8], [34,46], [72,80]m의 양쪽과 [0,40]m의 오른쪽 설치선을
금지합니다. 연결부는 station 30~50m입니다. `base`에는 이 추가 제한이 없습니다.
**모든 수치 임계값은 기능·민감도 실험용 예시입니다. 사용자가 현장 운영 기준을 정한 적은 없으며,
문헌도 이 숫자를 LiDAR 설치 표준으로 인증하지 않습니다.**

### 4. 도로 상대 전이와 곡선 연결

기존 T자/십자 전이는 동일 `(branch, side)` 안에서 설치 구간의 상대 좌표를 사용합니다.

$$
u=\frac{s-s_{\min}}{s_{\max}-s_{\min}},\qquad
u_*=(1-\alpha)u_0+\alpha u_1,\qquad
s_*=s_{\min,*}+u_*(s_{\max,*}-s_{\min,*}),\qquad
\alpha=\frac{w_*-w_0}{w_1-w_0}.
$$

횡편차·높이·pitch는 선형, 상대 yaw는 최단 원호로 보간합니다. 같은 설치선 안에서 Hungarian
대응 후 target 설치 격자로 투영하며 target coverage로 대응을 고르지 않습니다.
branch/side별 센서 수가 다르면 `unavailable`입니다. 이것이 24개 중 22개 보간 불가의 원인이었습니다.
이 경로의 자세 기준과 최신 corridor의 도로 **normal** 기준 상대 yaw는 구분합니다.

현재 연결 도로는 [ConnectedCorridorGeometry](src/environment.py)의 직선-원호-직선입니다.
총 길이 $L$, 원호 길이 $\ell$, 회전각 $\phi$를 **radian**으로 두면
$s_a=(L-\ell)/2$, 곡률 $\kappa=\phi/\ell$입니다. 원호 내부 $v=s-s_a$에서:

$$
c(s)=\left(s_a+\frac{\sin(\kappa v)}{\kappa},\,
                 \frac{1-\cos(\kappa v)}{\kappa}\right),\qquad
\theta(s)=\kappa v,\qquad
n(s)=(-\sin\theta(s),\cos\theta(s)).
$$

원호 이전은 $c(s)=(s,0)$, 이후는 원호 끝에서 방향 $\phi$의 직선으로 잇습니다.
$\kappa=0$이면 전체가 직선입니다. canonical pose $(s,d,z,\psi,\beta)$를 다음처럼 복원합니다.

$$
(x,y)=c(s)+d\,n(s),\qquad
d=\pm(w/2+w_{\mathrm{sidewalk}}/2),\qquad
\mathrm{yaw}=\mathrm{deg}(\theta)-90^\circ\operatorname{sign}(d)+\psi\pmod{360^\circ},\quad
\mathrm{pitch}=\beta.
$$

source 40m의 station을 첫 구간에는 그대로, 둘째 구간에는 +40m로 옮깁니다.
폭이 달라지면 설치선 offset을 target 폭으로 바꿉니다. 곡선과 함께 mesh·target·pose를 만들지만
**광선은 Euclidean 직선이며 구/원통 표면을 따라 휘거나 끝에서 반대쪽으로 넘어가지 않습니다.**
양 끝은 유한 경계이고 구형/원통형 주기 경계는 미구현입니다. ±90도 및 외측 offset보다 큰
곡률 반경을 요구하는 현재 구현 제한은 수치적 허용 조건이지 재사용 가능한 최대 곡률의 증명이 아닙니다.

최신 target 표본은 station/lateral 1m, 높이 1/1.5/2m이며 median을 제외합니다.
균일 $(s,d)$ 표본은 곡선의 면적 균일 표본이 아닙니다. 설치 후보는 station 4m마다 양측에
높이 3/5m × 상대 yaw -45/0/45도 × pitch -20/-10도 = 위치당 12개입니다.
base는 40위치·480 pose, combined는 18위치·216 pose입니다. 두 profile 평균이 348 pose입니다.
이것은 기존 100m 맵의 0.5m full-grid와 별개의 유한 격자입니다.

### 5. Marginal response와 source 재사용

순수 union의 추가 기여는 $\Delta_F(p\mid S)=F(S\cup\{p\})-F(S)$입니다.
상대 센서가 바뀌면 marginal response도 달라집니다. 일반적인 공통 목적값에 대해,
현재 센서 수를 $d_j(S)$, 추가 후보 mask를 $a_{pj}$라 두면:

$$
\Delta_J(p\mid S)=\sum_j(w_c+w_I\eta_j)a_{pj}\mathbf1[d_j(S)=0]
 -w_o\sum_j a_{pj}\mathbf1[d_j(S)\ge1]-w_n-w_b b(p).
$$

이 식은 목적값 변화이며 hard constraint 충족은 별도 검사입니다.
실행한 [source response 압축](src/methods/source_modes.py)은 빈 배치/다른 위치 센서 하나를
context $S$로 두고 위치별 후보군 $A$의 손실
$\ell_S(A)=\max_{p\in\mathcal C_h}\Delta_J(p\mid S)-\max_{p\in A}\Delta_J(p\mid S)$를 계산합니다.
단독 최고 후보에서 시작해 $\max_S\ell_S(A)$를 가장 줄이는 후보를 순차 추가하고,
동률은 평균 손실·단독 J·ID로 처리합니다. exact minimax 압축이나 학습된 response 모델은 아닙니다.

별도 [가림 응답 전이](src/methods/response_transfer.py)는
$B_s=g_s\land\neg a_s$, $\widehat a_t=g_t\land\neg\mathrm{map}(B_s)$를 사용합니다.
source에서 가린 표본을 target 도로 상대 좌표로 옮기는 heuristic이며 true target mask가 아닙니다.
잘못된 가림 전이로 유효 후보를 버릴 수 있어 최종 배치에는 실제 target raycast만 사용합니다.
순수 비음수 union의 diminishing-return 성질을 근거로, 중복 하한·비용·고장 제약을 포함한
전체 문제에 같은 greedy 근사보장을 적용하지 않습니다. marginal 계산 자체를 새 이론으로 주장하지 않습니다.

### 6. 최신 geometry-only 후보 선별과 정확한 pair 선택

설치 지점 $h=(s,\mathrm{side})$별로 실제 조회 예산 $q=1,2,4$를 둡니다.
예측 mask를 $g_c$, 실제 조회 mask를 $a_c$, 기하 단독 점수를
$s(c)=\sum_j(w_c+w_I\eta_j)g_{cj}-w_n-w_b b(p_c)$로 둡니다.
아래 $J(c,d;m_c,m_d)$는 지정 mask를 넣은 공통 목적값이고, 파트너는 다른 설치 지점만 허용합니다.

| 방법 | 질의할 pose를 정하는 규칙 |
|---|---|
| `static` | 지점별 $s(c)$ 상위 q개를 사전에 고정 |
| `random` | 지점별 복원 없는 균등 추출; seed별 전체 순열의 앞 q개, 관측 피드백 없음 |
| `static_pair` | $s_{pair}(c)=\max_d J(c,d;g_c,g_d)$, 예측 제약을 만족하는 파트너 중 최고값으로 지점별 순위 고정 |
| `adaptive` / `adaptive_batch` | 지점별 단독 최고값으로 시작; round r 이전 관측 집합 $Q_{r-1}$에서 $s_r(c)=\max_{d\in Q_{r-1}}J(c,d;g_c,a_d)$로 다음 조회 선택 |
| `no_feedback` | 같은 초기화·round·파트너·제약·동점 규칙에서 $a_d$ 대신 $g_d$ 사용; **모든 질의 계획을 첫 raycast 전에 고정** |
| `full` | 모든 설치 허용 pose를 조회; source-free full reference이며 Gurobi 호출 없음 |

예측 파트너가 없으면 단독 기하 점수로 fallback합니다. pair 순위는 소수점 8자리 반올림 점수,
단독 prior, candidate ID로 동점을 해소하고, round 내부에서는 이전 관측 context를 고정합니다.
`static_pair`의 최고 파트너가 실제 최종 질의 pool에 포함된다는 보장은 없습니다.
이는 joint query-pool 최적화가 아니라 선별 heuristic입니다. scalar/batch는 동일 결정 규칙을
다르게 계산하며 batch는 packed 교집합과 필요한 제약 검사를 묶어 CPU 비용을 줄입니다.

실제 조회한 후보 집합 $Q$가 고정되면 모든 방법의 최종 선택은 다음과 같습니다.

$$
P_Q=\mathop{\arg\max}_{\{c,d\}\subseteq Q,\;\mathrm{feasible}(c,d;a_c,a_d)}
                         J(c,d;a_c,a_d).
$$

[exact_pairs](src/methods/connected_pairs.py)는 모든 해당 쌍을 고려하되, 개선 불가능 점수나
필요조건으로만 가지치기하고 공통 evaluator로 최종 검사합니다. $Q=\mathcal C$일 때만
해당 **유한 격자·정확히 2대·선언한 제약**의 기준값 $J^*$입니다.
제한 pool의 optimum은 전역 상한이 아닙니다. pool에 feasible pair가 없으면 `null`이며,
전체 후보에서도 불가능하다는 증명이나 coverage 0점으로 바꾸지 않습니다.

### 7. LS와 MAPPO의 범위

LS는 한 번에 센서 하나의 이웃 이동을 평가하는 deterministic best-improvement 국소 탐색입니다.
기존 prototype의 XY 11-action·초기 XY 반경 10m와 corridor의 station/z/yaw/pitch 이웃·측면 고정·
station 반경 8m는 서로 다른 행동공간입니다. hierarchy 실험에는 지점/side를 고정하고 자세만
보정하는 대조군도 있습니다. 조회·평가·시간 제한 또는 개선 가능한 이웃 없음에서 종료합니다.

제약부 MAPPO의 유효 joint action 보상은 $r_t=J(P_{t+1})-J(P_t)$입니다.
위반 joint action은 전체 취소하고 raw reward -1을 주며, 뽑은 action/log-probability를 그대로 기록합니다.
이후 PPO buffer에는 기존 scale/clip을 적용합니다. 과거 무가중 v3의 거리·unique·outward shaping과
다른 경로이므로 결과를 섞지 않습니다. 할인·clipping 이후 보상이 최종 J와 정확히 동치인 것도 아닙니다.
정책/optimizer는 초기값마다 새로 만들고 **best feasible deployment와 final policy를 별도 저장**합니다.
최신 source-free corridor에는 LS/MAPPO를 사용하지 않았습니다.

<a id="progress-history"></a>

## 전체 진행 이력과 결과 해석

표의 횟수는 해당 runner의 배치/worker/slot 단위입니다. 서로 다른 실행을 더해 독립 지도 수나
통계적 표본 수로 쓰지 않습니다. 각 자료 링크와 뒤의 날짜별 절에 원 실행 명령·snapshot이 있습니다.

| 단계 / 날짜 | 완료 범위 | 핵심 결과와 남은 문제 | 원자료 |
|---|---|---|---|
| 전체 도로 MAPPO v3 / 09-28 | 180 runs | 36조건 평균 최고 deployment coverage 86.836%; 정책 수렴·전이 증거 아님 | [요약](results/mappo_v3_official_20260928_130911/V3_RESULTS_SUMMARY.md) |
| 폭 전이·보간 / 10-03 | 15맵, MAPPO 135회, 기준선 30개 | 보간의 일관된 단독 전이 우위 없음; T/십자 high는 배분 제약부 보조 source | [표](results/prototype_width_sweep_20261003/summary.csv) |
| 비대칭 형상 / 10-07~08 | 4맵, MAPPO 4,096 steps, 기준선 8개 | MAPPO best는 4맵 모두 top-10 MILP보다 낮음; 후보 공간도 다름 | [비교](results/prototype_shape_baselines_20261008/method_comparison.csv) |
| 신규 library / 10-08 | source 4개 × K1~6; target 72 slots | 전이 48·보간 2개 완료, 보간 불가 22개. 모든 source가 인증 최적은 아님 | [manifest](results/prototype_library_shapes_20261008/manifest.json) |
| 동일 초기값 보정 / 10-11~12 | 초기값 18개, LS18 + MAPPO54 | 평균 coverage LS97.642% > MAPPO97.087%; 기록 시간 3.91초 vs61.18초 | [paired](results/prototype_refinement_paired_20261011/paired_summary.csv) |
| target full-grid / 10-12 | 4맵 × K3,4 = 8개 | 모두 시간 제한, 상대 gap0.061~6.064%; 온라인 72배치의 archive 포함·density 일치 | [집계](results/prototype_full_grid_20261012/summary.csv) |
| 제약·복구 민감도 / 10-20 | 27 paired 반복 처리, 36 method 실행 | feasible 초기값 18회 확보, 9회 미실행; 짧은 128-step MAPPO이며 LS 우세 | [보고서](results/constrained_sensitivity_pilot_20261020/REPORT.md) |
| RSU v2 제약 / 10-20 | T/십자 K4, LS2 + MAPPO2 | MAPPO각4,096steps/16updates; 두 지도 모두 J는 LS보다 낮음 | [원자료](results/rsu_coverage_development_20261020/raw_methods.csv) |
| 연결 도로 / 10-20 | 7각도 × 6방법 = 42배치 | 공동 선택5/7 기준값 일치; 연결부는 이미 전부100%, 최대 재사용 각도 미확인 | [보고서](results/connected_corridor_pilot_20261020/REPORT.md) |
| 곡률·대안 수 / 10-21 | 36조건 × 6방법 = 216실행 | 후보1/2/4개 확장만으로 누락된 유용 자세 문제 해소 못함 | [분석](results/connected_curvature_sweep_20261021/analysis/REPORT.md) |
| 제약·다양성 / 10-21 | 개발16조건·112배치, 새24조건·168배치 | 다양성 조합16/16 일치가 새 조건에서는15/24; LS후18/24 | [새 조건](results/connected_constraint_holdout_20261021/analysis/REPORT.md) |
| 실패 진단·response / 10-22 | 48진단, 별도216배치 및 audit | 현재 제한 영역에서 LS 손실0; response도 LS후18/24로 동일 | [진단](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/FINDINGS.md), [audit](results/connected_response_study_20261022/validation_audit_20261022/FINDINGS.md) |
| 설치 배분/자세 분리 / 10-22 | 24조건·480출력 | 조건부 선별24/24 일치지만 source의 효과인지 분리 필요 | [요약](results/connected_hierarchy_20261022/summary.json) |
| prior ablation / 10-23 | 온라인504 + 기준선24 | source는 geometry 대비 q1에서16패, q2에서8패, q4모두동률 | [요약](results/connected_prior_ablation_20261023/summary.json) |
| source 가림 전이 / 10-23 | 온라인288 + 기준선24 | q2에서3개 개선, q4에서3개 악화; false-blocked40,998항목 발생 | [요약](results/connected_response_transfer_20261023/summary.json) |
| source-free / 11-03 | 180 workers | q4는12/12 기준값 일치, full 대비 시간 우위 없음 | [보고서](results/geometry_placement_20261103/REPORT.md) |
| 동일 결정 batch / 11-03 | 108 workers | scalar 대비 선별 단계 평균30.6% 감소; full보다는 느림 | [보고서](results/geometry_batch_20261103/REPORT.md) |
| 동일 pose 예산 / 11-04 | 264 workers | q4 기준값 일치 adaptive12/12, static4/12, random7/60 | [보고서](results/geometry_budget_20261104/REPORT.md) |
| 피드백 분리 Gate 1 / 11-09 | 468 workers | adaptive/no-feedback36/36 같은 배치·J; 강한 pair 정적 기준선 필요 | [보고서](results/geometry_mechanism_20261109/REPORT.md) |
| 인프라 문헌 검토 / 11-09 | 설계 문서, 실험 없음 | route 서비스·scenario·공유 인프라를 제안; 수치 요구값 미정 | [검토](RSU_INFRASTRUCTURE_REVIEW_20261109.md) |
| ARS 연구 평가 / 11-16 | 5관점 검토, 실험 없음 | 재현 가능한 개발 근거는 확보; 핵심 bound·도로망·시간 이점 미입증 | [평가](results/ars_review_20261116/REPORT.md) |

최근 구현은 source-free가 주 경로입니다. 원래 library·전이·MAPPO는 삭제하거나 성공 사례만
선별하지 않고 보존합니다. 단독 점수 prior의 실패를 모든 library 연구의 불가능성으로,
현재 피드백 동률을 모든 복잡한 장면에서의 무효성으로 일반화하지 않습니다.

<a id="research-review"></a>

## 연구 검토와 현재 결론 (2026-11-16 평가 반영)

[ARS-Codex 3.24.2 평가](results/ars_review_20261116/REPORT.md)는 연구 기여·방법론·LiDAR domain·
인프라 서비스와 반론 관점에서 계획과 개발 자료를 검토했습니다. 내부 결과 `major_revision`은
**상당한 연구 보강 필요**라는 개발 평가이며 실제 저널의 결정·게재 확률·T-ITS 적합성 인증이 아닙니다.
동일 모델 계열의 분리된 5개 검토이므로 독립적인 사람 심사와 같지 않습니다.
`NOT_CALIBRATED`, 정식 venue criteria binding 미확보 상태입니다. 당시 10개 입력 hash는 불변이었습니다.

| 확인 사항 | 해석 / 문서 보완 |
|---|---|
| 원자료468개 → 조건/방법156개·대조108행 재집계 일치 | 수치 보고의 일관성. 모든 물리 계산을 다시 수행했다는 뜻 아님 |
| 관측 피드백 추가 이점 없음 | 이미 검사한 부정 결과이며 단순히 "아직 비교하지 않음"으로 남기지 않음 |
| 가림 없는 예측 중 mesh에서 사라지는 항목 0.024949~3.370752% | 이번 장면의 제한된 optical/LOS 차이. 피드백 동률의 인과 증명 아님 |
| `result.work`는 최종 독립 raycast 이전 snapshot | 최신 표의 pose/ray 수는 **선택 단계**; 검산을 포함한 총 작업량 아님 |
| 작은 후보군 optimum과 full-domain bound는 다름 | 미래 인증은 실제 반환 incumbent와 동일 후보 공간의 유효 상한을 연결해야 함 |
| geometry 연결과 인프라 연결은 다름 | 공통 전원/데이터 경로·수용량·고장 후 재할당 정책은 별도 모델 필요 |
| 새로운 핵심 방법·전역 K·외부 검증 부재 | 노벨티는 미확정. 단순 구성요소 결합·제약 추가·MAPPO 사용만으로 성립하지 않음 |

가림 비율의 분모는 모든 target이나 ray 수가 아니라 **range/FOV가 참인 candidate-target 항목**입니다.
[저장 진단](results/geometry_mechanism_checks_20261109/feedback_diagnostics.json)에서 사후 계산했으며
선별 입력에 제공하지 않았습니다. 작은 비율이어도 특정 제약을 좌우할 수 있습니다.
검토 형식 검사 통과는 [validation.json](results/ars_review_20261116/validation.json)에 기록돼 있지만
논문 주장 전체의 과학적 인증은 아닙니다. 이번 README 갱신에서 새 심사를 수행하지 않았습니다.

<a id="proposed-model"></a>

## 미구현 연구 설계와 다음 검증

이 절은 **구현 결과가 아니라 향후 설계**입니다. 기존 실험의 J·제약·원자료에 소급 적용하지 않습니다.
원 계획은 [TITS_RESEARCH_PLAN.md](TITS_RESEARCH_PLAN.md), 인프라 제안의 문헌별 근거·열람 범위는
[RSU_INFRASTRUCTURE_REVIEW_20261109.md](RSU_INFRASTRUCTURE_REVIEW_20261109.md)에 있습니다.
두 문서의 과거 수치 평점은 계획용 판단이며 객관적 논문 점수나 게재 확률로 사용하지 않습니다.

### Partial Visibility Bound: Gate 2

geometry/설치 규칙으로 전체 후보 영역을 먼저 고정하고, 미조회 incidence를
$a^-_{ij}\le a_{ij}\le a^+_{ij}$로 감쌉니다. range/FOV 밖은 양쪽0,
조회 완료 항목은 양쪽이 실제 mask, 미조회 optical-positive 항목은 우선 [0,1]입니다.
source 전이·학습 confidence를 인증된 incidence bound로 자동 취급하지 않습니다.

최대화 문제의 검증된 반환 배치 $P_{\mathrm{returned}}$와 유효한 전역 상한 $U$가 있을 때:

$$
L=J(P_{\mathrm{returned}})\le J^*\le U,\qquad
0\le J^*-J(P_{\mathrm{returned}})\le U-L.
$$

다른 배치에서 얻은 L을 쓰려면 적어도 $J(P_{\mathrm{returned}})\ge L$이어야 합니다.
그렇지 않으면 반환 배치의 bound는 $U-J(P_{\mathrm{returned}})$입니다.
feasible incumbent가 없으면 품질 gap을 인증하지 않습니다. 이 부등식은 표준 상·하한 논리이지
새 정리가 아닙니다. 구현할 **구체적인 저비용 상한 구성과 질의 규칙은 TBD**입니다.

주의할 점은 $J$에 중복 패널티가 있어 $J(P;a^+)$를 그대로 상한으로 둘 수 없다는 것입니다.
예를 들어 두 센서가 하나의 target을 모두 본다는 예측은, 실제로 한 센서만 볼 때보다
패널티 때문에 J가 작을 수 있습니다. 중복 하한·상한과 N-1도 각각 별도 interval 논리가 필요합니다.
작은 문제에서는 가능한 incidence 완성·후보 부분집합을 전수조사해 다음을 나눠 검사해야 합니다.

1. 상한의 유효성: true finite optimum을 절대 밑돌지 않는가.
2. 유용성: 미해결 정보의 불확실성, relaxation의 느슨함, 약한 incumbent를 분리했는가.
3. 비용: bound 생성·query 선택·최종 검산을 더해도 전체 visibility 계산보다 유리한가.

낙관적 incidence 완성이 실제 도로 mesh로 실현 가능해야 하는지, 추상 binary 완성까지 허용하는지도
명시해야 합니다. 작은 exact optimistic-completion 비교부터 수행하고, 유효하지만 항상 너무 큰
상한이거나 계산이 비싸면 큰 도로망 실행 전에 설계를 재검토합니다. 현재 Gate 2 성능표는 **TBD**입니다.

### 도로망과 RSU 인프라 확장

도로 graph(경로/분기), optical visibility relation(인접하지 않은 도로 사이 관측 포함),
전원/데이터 graph(공유 경로·용량·고장)를 구분합니다. global target ID와 물리 센서 ID를 유지해야
모듈 coverage를 중복 합산하거나 고장 센서를 중복 백업으로 세지 않습니다.
전체 K 할당과 공유 설치비가 결합된 최적화는 아직 corridor의 K=2 전수조사를 넘어 구현되지 않았습니다.

후보 선택 $x_i$, site 활성화 $y_h$, 인프라 link 활성화 $z_e$, 명시한 scenario $\omega$에 대해:

$$
d_{j,\omega}=\sum_i x_i\,b_{i,\omega}\,a_{ij,\omega},\qquad
B=\sum_i c_i x_i+\sum_h c_h y_h+\sum_e c_e z_e+B_{\mathrm{O\&M}}.
$$

$b_{i,\omega}$는 해당 센서가 필요한 전원과 지정 데이터 sink까지의 서비스를 확보했는지입니다.
공유 자산은 한 번만 과금하고 기존 자산과 신규 투자비를 구분하는 제안입니다.
고장 전 고정 경로인지, 고장 후 재할당을 허용하는지와 그때의 용량도 정해야 합니다.
현재 수평 hub 매칭이 이 graph 모델을 이미 구현한 것은 아닙니다.

경로/ROI r에서 coverage $C_{r,\omega}$와 최장 연속 미관측 길이 $G_{r,\omega}$를 정의해
예산 $B\le B_{\max}$ 아래 coverage를 최대화하거나, 서비스 하한을 만족하는 최소 비용을 구할 수 있습니다.
두 문제 중 실제 계획 의사결정을 먼저 선택해야 하며 운영 임계값은 아직 없습니다.
현재 `max_sampled_blind_run_m`는 고정 lateral/height track의 연속 미관측 station-bin 수 × 간격입니다.
임의 차로 경로·비균일 표본·교차로 통과·차량 추적을 처리하는 서비스 계약은 아닙니다.

Wi-Fi/중계기에서 가져올 수 있는 것은 평균+최악 구간 서비스, 접근로별 하한, shared failure,
공유 설치비 구조입니다. SINR·RF 간섭·packet delay의 수식/임계값을 optical coverage에 대입하지 않습니다.
통신 RSU와 roadside LiDAR는 동일 장비가 아니며 공동 site/전원/backhaul을 사용할 수 있는 관계입니다.
추가 제약은 LS·MILP·MAPPO에 공통 적용해야 하며 MAPPO 우위를 미리 가정하지 않습니다.

| Gate | 상태와 다음 완료 기준 |
|---|---|
| 1. 기여 분리 | 완료: pair 정적/no-feedback 비교와 부정 결과 보존 |
| 2. bound | 미구현: 작은 exact 문제에서 유효성·tightness·총 비용 검증 |
| 3. 도로망 | 미구현: 다중 연결·가변 K·global target/센서 ID·물리 상호작용·인프라 충돌 |
| 4. 규모 | 미완료: 후보/target/mesh/도로망 크기를 따로 변화시킨 quality-query/time 곡선 |
| 5. 외부 근거 | 미완료: 방법 고정 후 미사용 geometry 가족·실제 자산·정당화한 서비스 조건 |
| 6. 논문 | 미완료: 가까운 선행방법별 차이, 주장-근거 대응, 한계와 재현 자료 |

현재 개발 사례는 부정 대조군으로 남깁니다. MAPPO 학습량 증가나 제약 개수 추가만으로
후속 검증을 대신하지 않습니다. 검출·추적은 계속 후속 범위입니다.

<a id="current-reproduction"></a>

## 재현 경로와 최신 파일 책임

모든 명령은 프로젝트 root에서 실행합니다. 이 작업 환경의 실제 root는
`/home/robotserver/hj_ws_main`이며 과거 JSON의 절대 경로는 자동 수정하지 않았습니다.
새 실행은 새 output 이름을 사용합니다. 아래 첫 명령은 파일·raycast·학습을 만들지 않습니다.

```bash
.venv/bin/python geometry_placement.py \
  --methods adaptive_batch static static_pair no_feedback full \
  --query-budgets 1 2 4 --repeats 3 --output results/geometry_mechanism_repeat
```

실제 재실행은 위 명령에 `--execute`를 추가합니다. 원 실행의 환경 변수와 명령은
[아래 Gate 1 절](#mechanism-results)에 있습니다. 당시 조건을 재현하려면 현재 소스만이 아니라
`results/geometry_mechanism_20261109/source/`와 `inputs/`의 snapshot/hash를 확인해야 합니다.
report 명령은 기존 실험을 다시 풀지 않지만 파생 CSV/JSON을 갱신하므로 읽기 전용 검산과 구분합니다.

| 책임 | 구현 / 명세 |
|---|---|
| 공통 geometry·설치·각도 | [environment.py](src/environment.py), [visibility.py](src/core/visibility.py) |
| J·비용·제약 평가 | [evaluation.py](src/core/evaluation.py), [coverage_requirements.py](src/core/coverage_requirements.py), [corridor_requirements.py](src/roadside/corridor_requirements.py) |
| 후보 budget·optical 예측·조건부 선택 | [connected_hierarchy.py](src/methods/connected_hierarchy.py) |
| 사전 계획 pair/no-feedback 대조군 | [geometry_controls.py](src/methods/geometry_controls.py) |
| 유한 K=2 기준선·corridor LS | [connected_pairs.py](src/methods/connected_pairs.py) |
| fresh worker·계측·CLI | [geometry_placement.py](geometry_placement.py), [connected_corridor_study.py](connected_corridor_study.py) |
| 최신 보고서·검산 | [geometry_mechanism_report.py](geometry_mechanism_report.py), [validate_geometry_placement.py](tests/validate_geometry_placement.py) |
| source pose / response 전이 | [prototype_segments.py](prototype_segments.py), [source_modes.py](src/methods/source_modes.py), [response_transfer.py](src/methods/response_transfer.py) |
| 제약부 초기 복구·LS/MAPPO | [feasibility_repair.py](src/methods/feasibility_repair.py), [constrained_refinement.py](src/methods/constrained_refinement.py) |
| 예전 full-grid packed 기준선 | [packed_candidates.py](src/roadside/packed_candidates.py), [coverage_cut_milp.py](src/methods/coverage_cut_milp.py) |

**시간 경계:** `decision_sec`는 자식의 입력/geometry/scene/선별/조합,
`cold_ready_sec`는 부모 process 시작 직전부터 자식 선택 완료까지(import 포함),
`worker_wall_sec`는 독립 검산·저장·process 종료까지입니다. 별도 audit의 full-mask 재계산은
온라인 latency가 아닙니다. `result.work`/표의 ray 수는 선택 단계 snapshot이므로 총 검산 ray로 읽지 않습니다.
세 번 반복은 case별 median/IQR로 요약하고, 그 뒤 case를 동일 가중 평균합니다.
seed·좌우 반사·profile·timing repeat를 독립 지도 표본으로 세지 않습니다.

경량 확인 명령은 아래와 같습니다. 테스트의 임시 raycast/PPO는 정식 실험과 다릅니다.

```bash
.venv/bin/python -m py_compile main.py test.py view_result.py src/environment.py \
  src/outputs.py src/visualize.py src/methods/milp.py src/methods/greedy.py \
  src/methods/mappo.py tests/test_pipeline.py geometry_placement.py \
  geometry_mechanism_report.py src/methods/geometry_controls.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_pipeline.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_geometry_controls.py
```

날짜별 절의 75/143 tests 등은 **해당 시점의 검증 기록**이며 서로 더해 현재 테스트 수로 쓰지 않습니다.
**이번 갱신에서 확인한 범위:** pipeline 7개와 geometry control 6개, 합계 **13 tests 통과**;
위 13개 Python 파일 문법 검사, 새 geometry 계획과 기존 MILP/GREEDY/MAPPO dry-run 3개 통과입니다.
새 표의 **수치 96셀**을 원자료와 대조하고, 저장 worker 468개 JSON/시간과 CSV 156개 집계의 일치를
다시 확인했습니다. README 로컬 링크·명시적 anchor·code/math fence·공백 검사도 통과했습니다.
실행 영수증은 [checks.json](results/readme_update_20261118/checks.json), 문서 대조는
[documentation_checks.json](results/readme_update_20261118/documentation_checks.json)에 보존합니다.
이 검증은 저장 결과의 산술/출처 확인과 소형 회귀이며, 468개 정식 실험·전체 visibility·대규모 solver·
정식 학습을 다시 실행하거나 모든 과거 수치의 물리 검산을 반복한 것이 아닙니다.

## 이전 Prototype 결과 안내 (2026-10-14 기준 기록)

아래 블록은 기존 pose 전이 연구의 결과 경로 안내입니다. 최신 geometry-only 연구와 목적함수·후보 영역이 다릅니다.

> 당시 문서 확인일: **2026-10-14**. 최신 완료 라이브러리: `results/prototype_library_shapes_20261008/`.
> 동일 초기값 보정 비교: `results/prototype_refinement_paired_20261011/` 72/72 완료.
> 최신 전체 격자 기준선: `results/prototype_full_grid_20261012/` **8/8 완료**, 2026-10-12 22:04:28 KST.
> 모두 시간 제한 해이며 상대 gap 0.061~6.064%. 아래 새 비교표는 온라인 배치 72개의 archive 포함/관측 밀도 일치를 검증했습니다.
> 이전 대표 격자 기준선: `results/prototype_cut_study_20261012/` 8/8 완료. 그 공간에서만 2개 최적성 확인, 나머지 gap 0.079~1.477%.
> 이전 `results/prototype_target_full_20261012/`는 cache/후보 archive 및 과거 기준선입니다. 7개가 presolve 시간 초과였으므로 최신 성능표와 구분합니다.
> 새 library 4개 × K=1~6 저장 완료. 새 지도 단독 전이 48개, 보간 2개 완료 / 22개 대응 불가. 이번 실행에는 MAPPO 없음.
> 이전 폭 변화 비교: `results/prototype_width_sweep_20261003/` (15개 맵, MAPPO 135회, 기준선 30개).
> 추가 형상 GREEDY/MILP 기준선 완료: `results/prototype_shape_baselines_20261008/` (8개 실행, 재검산 8/8 통과).
> 추가 형상 개발 파일럿: `results/prototype_shape_pilot_20261007/`. 기존 폭 실험과 구분합니다.
> 같은 형상의 PPO 학습량 확장: `results/prototype_shape_ppo4096_20261007/` (4개 맵, 4,096 steps, seed=0).
> 기존 전체 도로용 MAPPO v3 결과: `results/mappo_v3_official_20260928_130911/`. 최신 국소 보정과 다른 실험입니다.
> 과거 `journal_r30*` 결과는 삭제된 과거 실험이며 현재 결과와 혼용하지 않습니다.
> 생성 결과는 ignored 경로이므로 저장소 clone만으로 포함되지 않습니다.
> 기존 세미나와 `seminar_k11_20260921`은 MAPPO v2 결과입니다. v3는 2026-09-28에 180/180 runs를 완료했습니다. v2와 v3 및 UI 예제의 수치를 구분하세요.

<a id="mechanism-results"></a>

## 쌍 기반 기준선 및 피드백 분리 검증 (2026-11-09)

T-ITS 연구 계획의 1단계를 구현하고 **468/468 작업**을 완료했습니다.
기존 개발 형상 6개 × base/combined의 12조건, 고정 K=2, 위치당 조회 예산 1/2/4,
4개 선별 방법 및 full 기준선, 시간 반복 3회입니다. source library·MAPPO·검출은 사용하지 않았습니다.
반복은 시간 측정용이며 독립 지도나 성능 표본이 늘어난 것은 아닙니다.

- `static_pair`: 모든 후보를 다른 위치의 예상 파트너와 조합해 점수를 매긴 뒤 위치별 순위를 고정합니다.
- `no_feedback`: adaptive와 같은 초기화·라운드·제약·동점 규칙을 쓰되 파트너의 실제 관측값 대신
  가림 없는 geometry 예측을 사용합니다. **첫 실제 raycast 전에 전체 질의 계획을 확정**합니다.
- 두 방법 모두 최종 센서 선택은 실제 조회한 mask로 하고, 저장 배치는 별도 direct raycast로 검산합니다.
  가상 마스크·쌍 평가량과 실제 pose/ray 질의량을 구분합니다. 기존 방법과 기본 CLI는 유지했습니다.

전체 허용 후보의 이산 K=2 최적값 대비 **평균 목적값 J 손실**입니다. 낮을수록 좋으며 coverage %p가 아닙니다.

| 위치당 조회 예산 | Adaptive | 기존 단일 정적 | 쌍 기반 정적 | 피드백 제거 |
|---:|---:|---:|---:|---:|
| 1 | 7.075 | 7.075 | 4.925 | 7.075 |
| 2 | 1.183 | 6.675 | 0.525 | 1.183 |
| 4 | 0.000 | 4.342 | 0.000 | 0.000 |

같은 조건의 **full 유한 기준값과 J 일치 수 / 12조건**은 다음과 같습니다.

| 위치당 조회 예산 | Adaptive | 기존 단일 정적 | 쌍 기반 정적 | 피드백 제거 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 6 | 2 |
| 2 | 6 | 2 | 10 | 6 |
| 4 | 12 | 4 | 12 | 12 |

예산 2에서 차이가 생기는 조건을 포함한 전체 조건별 J입니다. 폭/각도/원호 길이는
[고정 입력](configs/geometry_placement.json)이며, 좌/우 반사는 독립 geometry 가족으로 세지 않습니다.
마지막 두 열은 `full J - method J`입니다. 예산 4에서는 static을 제외한 세 방법의 차이가 모두 0입니다.

| target: 폭 / 회전각 / 원호 길이 | profile | Full J | Adaptive q2 손실 | Static-pair q2 손실 |
|---|---|---:|---:|---:|
| 9m / -40도 / 18m | base | 1888.35 | 2.95 | 0.00 |
| 9m / +40도 / 18m | base | 1888.35 | 2.95 | 0.00 |
| 11m / -65도 / 22m | base | 2357.00 | 3.15 | 3.15 |
| 11m / +65도 / 22m | base | 2357.00 | 3.15 | 3.15 |
| 13m / -80도 / 27m | base | 2837.55 | 0.00 | 0.00 |
| 13m / +80도 / 27m | base | 2837.55 | 0.00 | 0.00 |
| 9m / -40도 / 18m | combined | 1871.95 | 0.00 | 0.00 |
| 9m / +40도 / 18m | combined | 1875.00 | 0.00 | 0.00 |
| 11m / -65도 / 22m | combined | 2352.45 | 0.00 | 0.00 |
| 11m / +65도 / 22m | combined | 2352.45 | 0.00 | 0.00 |
| 13m / -80도 / 27m | combined | 2825.50 | 1.00 | 0.00 |
| 13m / +80도 / 27m | combined | 2825.50 | 1.00 | 0.00 |

**핵심 해석:** adaptive와 피드백 제거의 최종 센서 ID·목적값은 36개 조건/예산 조합 모두 같습니다.
질의 후보군은 예산 2에서 8/12, 예산 4에서 10/12가 달랐습니다. 피드백이 작동하지 않은 것이 아니라,
질의 변경이 이번 조건의 최종 성능 이점으로 이어지지 않았습니다. 이전 단일 정적 대비 우위를
관측 피드백 자체의 효과라고 설명하지 않습니다.

예산 2의 쌍 기반 정적은 adaptive보다 4조건에서 높고 8조건에서 같습니다. full 목적값 일치는
10/12 대 6/12입니다. 반면 예산 1에서는 쌍 기반 정적이 기존 단일 정적보다 낮은 조건도 2개 있습니다.
예산 4에서는 adaptive·쌍 기반 정적·피드백 제거 모두 full과 12/12 일치합니다.

**시간 우위도 없음:** 예산 4의 평균 cold-ready는 각각 3.127/3.147/3.133초, full은 2.879초입니다.
조건별 3회 중앙값을 먼저 구해 12조건을 동일 가중치로 평균했습니다. 최종 검산·저장까지 포함한
worker 전체 평균도 각각 3.471/3.492/3.474초 대 full 3.231초로 같은 방향입니다.
질의 수는 평균 116 대 full 348이지만, 질의 절감만으로 end-to-end 시간 절감을 주장하지 않습니다.
공유 서버·OS cache 미격리 조건이고 실지도 일반화나 통계적 우위 검정이 아닙니다.

예산 4의 전체 방법·계측 경계를 함께 보면 다음과 같습니다. ray 수는 **선택 단계** 평균이며
독립 최종 검산/audit의 ray를 포함하지 않습니다. 시간은 case별 3회 중앙값의 평균입니다.

| 방법 | Pose 조회 | 선택 단계 ray 수 | Cold-ready, 초 | Worker 전체, 초 | Full J 일치 |
|---|---:|---:|---:|---:|---:|
| Adaptive batch q4 | 116 | 153921.25 | 3.127 | 3.471 | 12/12 |
| Static q4 | 116 | 164903.00 | 2.396 | 2.740 | 4/12 |
| Static-pair q4 | 116 | 151315.58 | 3.147 | 3.492 | 12/12 |
| No-feedback q4 | 116 | 150691.42 | 3.133 | 3.474 | 12/12 |
| Full | 348 | 402311.75 | 2.879 | 3.231 | 12/12 |

출처: [case/arm 비교표](results/geometry_mechanism_20261109/comparison.csv),
[집계 JSON](results/geometry_mechanism_20261109/mechanism_report/summary.json).
static은 더 빠르지만 품질이 다른 비교입니다. 저예산에서도 품질-시간 절충이 있으므로,
허용 J 손실을 미리 정하지 않고 최저 시간만 골라 "더 효율적"이라고 보고하지 않습니다.

검증: **75 tests**, 문법/기존 CLI dry-run, 독립 후보군 최적값 156개, 정책 재생 144개,
468개 배치 직접 검산, fresh visibility 6,912,000항목 및 독립 삼각형 표본 816건 통과.
예산별 후보 포함/품질 비감소 288쌍, 예산 1 초기화 일치 36쌍, 별도 산술 집계도 통과했습니다.
기존 방법의 84개 조건/방법을 새 252 workers와 대조해 결정·mask·최종 배치가 같았고,
기존 파일 6,207개의 해시를 보존했습니다. 정책 재생은 생산 코드를 공유하며 표본 LOS 검사는 전수 증명이 아닙니다.

```bash
# 새 root 필요. --execute 없으면 계획만 확인합니다.
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py \
  --methods adaptive_batch static static_pair no_feedback full \
  --query-budgets 1 2 4 --repeats 3 \
  --output results/geometry_mechanism_20261109 --execute
.venv/bin/python results/geometry_mechanism_20261109/source/geometry_mechanism_report.py \
  results/geometry_mechanism_20261109
```

[방법·고정 프로토콜](GEOMETRY_MECHANISM_STUDY.md),
[전체 결과](results/geometry_mechanism_20261109/REPORT.md),
[조건별 목적값](results/geometry_mechanism_20261109/mechanism_report/case_objectives.csv),
[쌍별 차이 108행](results/geometry_mechanism_20261109/mechanism_report/paired.csv),
[피드백 진단](results/geometry_mechanism_checks_20261109/feedback_diagnostics.json).
결과는 ignored 경로라 clone에는 포함되지 않습니다. 기존 실험 원자료는 수정하지 않았습니다.

**후속 결정:** 관측 피드백 우위를 핵심 기여로 확정하지 않고, 강한 geometry-only 기준선을 유지합니다.
다음은 작은 전수조사 문제에서 미계산 visibility의 하한/상한을 타당하고 저렴하게 구성할 수 있는지
검증하는 단계입니다. 쌍 점수에 새 이름을 붙이거나 학습량만 늘리는 것으로 노벨티를 주장하지 않습니다.

## 동일 Pose 조회 예산 선별 검증 (2026-11-04)

기존 개발 형상 6개 × base/combined 제약의 **12조건**, 고정 K=2에서 **264/264 실행 완료**했습니다.
`adaptive_batch`, 정적 단일 센서 기하 순위(`static`), 실제 무작위 후보군(`random`, seed 0~4)을
설치 가능 위치당 pose 조회 1/2/4회로 비교했습니다. 모든 방법의 실제 pose 조회 수가 일치하며,
전체 후보 `full`은 별도 평가 기준선입니다. 이번 실행은 source library·MAPPO·검출을 사용하지 않습니다.

**이전 random-prior와 구분:** 이전 실험은 adaptive의 초기 순서만 무작위였습니다.
새 `random`은 각 위치에서 복원 없이 균등 추출하고, 실제 관측이나 optical 점수로 다음 후보를 고르지 않습니다.
같은 seed에서는 예산 증가 시 기존 후보를 유지합니다. 무작위 5 seeds는 지도 5개가 아니며,
case 안에서 먼저 평균하고 12조건을 동일 가중치로 집계합니다. 가장 좋은 seed만 고르지 않습니다.

아래는 전체 허용 후보의 이산 K=2 최적값 대비 **평균 목적값 J 손실**입니다.
J는 coverage·중복·설치비용을 포함하므로 coverage %p나 상대 gap %가 아닙니다.

| 위치당 조회 예산 | 실제 pose 조회 수 평균 | Adaptive | 정적 기하 순위 | 무작위 5-seed 평균 |
|---:|---:|---:|---:|---:|
| 1 | 29 | 7.075 | 7.075 | 32.478 |
| 2 | 58 | 1.183 | 6.675 | 15.740 |
| 4 | 116 | 0.000 | 4.342 | 8.384 |

- 예산 1에서는 adaptive/static의 후보군이 12조건 모두 같습니다.
- 예산 2와 4에서는 각각 adaptive가 static보다 **8조건 우세·4조건 동률**, 열세 0건입니다.
- 예산 4의 전체 후보 최적값 일치는 adaptive **12/12**, static **4/12**, random **7/60**입니다.
  random의 60은 12조건 × 5 seeds이며 독립 지도 60개가 아닙니다.
- 무작위 seed 평균보다 adaptive가 각 예산의 12조건 모두 높았습니다. 개별 seed에서는
  예산 1에 2회, 예산 2에 1회 무작위가 더 높았으므로 모든 seed를 지배한다는 뜻은 아닙니다.
- **시간 우위는 없음:** 예산 4 adaptive 평균 cold-ready **3.124초**, full **2.884초**입니다.
  full은 평균 348 pose를 조회합니다. 이번 시간 반복은 1회이므로 정식 반복 speedup 검증은 아닙니다.
- pose 조회 수만 맞춘 비교입니다. 예산 4의 평균 실제 ray 수는 adaptive 153,921.25,
  static 164,903.00, random 133,645.03으로 다르며 optical 계산량도 별도입니다.

264개 모두 feasible입니다. 회귀 **69 tests**, 문법/CLI 계획 검사, 독립 유한 후보군 최적값 264개,
정책 재생 252개, fresh visibility 6,912,000항목, 독립 삼각형 LOS 표본 816건을 통과했습니다.
예산 증가 시 후보 포함 및 목적값 비감소 168쌍, 별도 평균/표본 SD 재계산도 통과했습니다.
기존 두 geometry 실행의 품질·배치 84개 대조가 일치하고 원자료 hash 2,669개를 보존했습니다.
이 표본 LOS 검사는 모든 광선에 대한 독립 구현 증명이 아니며, 정책 재생은 선별 코드를 공유합니다.

**해석:** 현재 조건에서는 두 단순 기준선보다 조회 예산을 효과적으로 사용했습니다.
그러나 adaptive는 관측 피드백뿐 아니라 센서 간 상호작용을 고려한 점수도 사용하므로,
이번 비교만으로 피드백 자체의 기여를 분리하거나 노벨티를 입증하지는 못합니다.
당시 후속 과제였던 상호작용 정적 기준선·피드백 제거 비교는 위 2026-11-09 절에서 완료했고,
관측 피드백의 추가 이점은 확인되지 않았습니다. 미사용 형상·가림 복잡도 검증은 여전히 남아 있습니다.
현재는 익숙한 개발 조건의 두 구간/K=2 결과이지 큰 도로망 최적화 검증이 아닙니다.

```bash
# --execute 없으면 계획만 출력. 기존 결과를 덮어쓰지 않으므로 재실행에는 새 root 사용.
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py --methods adaptive_batch static random full \
  --query-budgets 1 2 4 --random-seeds 0 1 2 3 4 --repeats 1 \
  --output results/geometry_budget_20261104 --execute
.venv/bin/python results/geometry_budget_20261104/source/geometry_budget_report.py \
  results/geometry_budget_20261104
```

[전체 보고서](results/geometry_budget_20261104/REPORT.md),
[조건별 비교 36행](results/geometry_budget_20261104/budget_report/paired.csv),
[집계 및 검증 범위](results/geometry_budget_20261104/budget_report/summary.json),
[독립 검산](results/geometry_budget_20261104/independent_validation.json).
실행 코드·입력·해시는 결과 root에 고정했습니다. 생성 결과는 ignored 경로로 clone에 포함되지 않습니다.

## 조건부 선별 Batch 최적화 (2026-11-03)

`--methods adaptive_batch`를 별도 추가했습니다. 기본 `adaptive`와 기존 실험 경로는 유지합니다.
라운드마다 실제 관측 mask를 한 번 모으고 packed-bit 교집합으로 후보/파트너의 필요 제약을
묶어서 검사합니다. 가능한 파트너가 여럿이면 **기존 dot product·8자리 반올림·ID 동점 규칙**을
그대로 사용하며, 최종 feasibility는 공통 evaluator로 확인합니다. 목적함수나 물리 모델을
근사로 바꾼 것이 아니며, 실제 조회 예산을 늘리거나 full cache로 몰래 대체하지 않습니다.

```bash
# 계획: raycast/파일 생성 없음. 실제 재실행에는 새 output 이름 사용
.venv/bin/python geometry_placement.py --methods adaptive adaptive_batch full \
  --query-budgets 4 --repeats 3 --output results/geometry_batch_repeat

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py --methods adaptive adaptive_batch full \
  --query-budgets 4 --repeats 3 --output results/geometry_batch_20261103 --execute
```

기존 개발 형상 6개 × base/combined × 3방법 × 3회로 **108/108 worker 완료**했습니다.
이번 실행 안의 scalar/batch 36쌍 모두 조회 순서·라운드별 파트너·예측 점수·최종 배치와 J가 같습니다.
표는 case별 3회 median을 12조건에 걸쳐 평균한 값이며, 선별 시간은 cold-ready의 일부입니다.

| 방법 | 평균 실제 조회 수 | 전체 유한 기준선 J 일치 | 조회/선별 단계 | Cold-Ready |
|---|---:|---:|---:|---:|
| Adaptive scalar / 4개 | 116 | 12/12 | 1.273초 | 3.548초 |
| Adaptive batch / 4개 | 116 | 12/12 | 0.883초 | 3.149초 |
| Full | 348 | 12/12 | 0.242초 | 2.893초 |

선별 단계 평균은 **30.6% 감소**했고, batch는 scalar보다 12조건 모두 빨랐습니다.
그러나 full보다도 12조건 모두 느리며, 같은 J의 case별 시간 비율 평균은 batch/full **1.087**입니다.
따라서 **같은 조회·배치의 CPU 비용 개선**은 확인했지만 full 대비 시간 우위는 아직 없습니다.
새 연구 알고리즘의 노벨티나 큰 도로망의 성능으로 주장하지 않습니다. 기본값은 바꾸지 않았습니다.

회귀 **143 tests**, 문법 검사, 기존 3방법 및 새 CLI 계획 확인을 통과했습니다.
검산은 108개 배치/clock, 36개 유한 후보군 최적값, 24개 정책 재생,
6,912,000개 fresh visibility 항목과 독립 삼각형 LOS 표본 816건을 통과했습니다.
이전 scalar/full 72개 실행과도 배치·선별 결과가 같고, 기존 기록 hash는 보존했습니다.
결과와 재현 범위는 [보고서](results/geometry_batch_20261103/REPORT.md),
[비교표](results/geometry_batch_20261103/comparison.csv),
[검산](results/geometry_batch_20261103/independent_validation.json),
[시간/단계별 비교](results/geometry_batch_20261103/paired_latency.json)를 봅니다.
결과는 ignored 경로이며, 아래 180회 실행의 과거 수치는 덮어쓰지 않았습니다.

## Source-Free Geometry 실행 및 시간 비교 (2026-11-03)

`geometry_placement.py`는 기존 source library·pose·visibility 파일을 요구하지 않습니다.
기본값은 **adaptive geometry / 지점·side당 4개 조회 / 반복 1회**입니다.
기본 실행에는 전체 후보 reference 생성도 없습니다. 최종 배치는 실제 조회 mask와 설치/중복/연결부
제약으로 검증하며, 전체 reference를 요청하지 않았을 때 gap을 0으로 표시하지 않습니다.

| 방법 | 후보 조회 방식 |
|---|---|
| `adaptive` (기본) | target range/FOV 초기 순위 + 이미 실제 측정한 상대 후보와의 조건부 J |
| `adaptive_batch` (명시적 요청) | 같은 조건부 선별의 packed 필요 제약 검사·라운드별 관측 재사용 |
| `static` | target range/FOV 단독 J 상위 후보를 지점/side별 고정 선별 |
| `full` (명시적 요청) | 설치 가능한 전체 후보를 실제 조회한 뒤 고정 K=2 전체 조합 선택 |

최종 조합은 모든 방법에서 같은 실제 mask와 제약을 사용합니다. source 단독 점수, 전이 mask,
LS/MAPPO, pretrained policy를 사용하지 않습니다. `full`은 이 유한 후보군의 K=2 기준선이지
연속 공간/도시 전체 최적해가 아니며 Gurobi를 호출하는 경로도 아닙니다.

```bash
# 계획만 출력: 파일/raycast 없음
.venv/bin/python geometry_placement.py

# source/reference 없이 기본 방법만 실행: 기존 output이 있으면 새 이름 사용
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py --output results/geometry_default_repeat --execute

# 비교: 기존 개발 형상 6개 x base/combined 2종 x 방법/예산 5종 x 3회 = 새 worker 180개
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python geometry_placement.py --methods adaptive static full \
  --query-budgets 2 4 --repeats 3 --output results/geometry_placement_20261103 --execute
```

각 조합은 새 CPU process와 비공유 raycast cache에서 실행합니다. 실행 순서는 case/repetition별
순환 배치이며, full reference가 앞서 실행되더라도 다른 worker에는 그 mask/배치/목적값을
입력하지 않습니다. 각 worker는 고정한 geometry/config와 자기 cache만으로 결정합니다.
OS 파일 cache를 지우거나 공유 서버 전체를 격리한 것은 아닙니다.

시간은 세 가지로 구분합니다.

- `cold_ready_sec`: 부모의 process 시작 직전부터 자식의 배치 선택 완료까지. interpreter/import 포함.
- `decision_sec`: 자식 내부 입력/config·geometry/scene·후보 예측/실제 조회·조합 시간. import 제외.
- `worker_wall_sec`: 최종 직접 검산과 배열 저장까지 포함한 process 전체 시간.

Linux 공통 `CLOCK_MONOTONIC`으로 부모/자식 경계를 대조하며, 최종 검산/export는 cold-ready 뒤입니다.
3회는 timing 반복이지 새 지도 3개나 최고 품질 선택이 아닙니다. case별 median/IQR과 동일 J 조건의
latency 비율을 구분해 보고하고, 품질이 낮은 해를 빠르다는 이유로 동일 품질 speedup으로 표시하지 않습니다.
기본 실행의 검산은 실제 조회한 pose만 다시 검사하며, 요청하지 않은 full cache를 몰래 생성하지 않습니다.

완료 여부는 [manifest](results/geometry_placement_20261103/manifest.json),
반복 원자료는 [runs.csv](results/geometry_placement_20261103/runs.csv),
조건별 비교는 [comparison.csv](results/geometry_placement_20261103/comparison.csv),
검산은 [independent_validation](results/geometry_placement_20261103/independent_validation.json)으로 확인합니다.
**180/180 worker 완료, 배치 180개 feasible, 회귀 135 tests 통과.** 다음 시간은
각 case의 3회 median을 12조건에 걸쳐 평균한 cold-ready 값이며 전체 CLI wall time이 아닙니다.

| 방법 | 평균 실제 조회 수 | 전체 기준선 J 일치 | Cold-Ready |
|---|---:|---:|---:|
| Adaptive / 2개 | 58 | 6/12 | 2.71초 |
| Adaptive / 4개 (기본) | 116 | 12/12 | 3.55초 |
| Static / 2개 | 58 | 2/12 | 2.37초 |
| Static / 4개 | 116 | 4/12 | 2.41초 |
| Full | 348 | 12/12 | 2.90초 |

Adaptive 4는 조회 수를 **1/3**로 줄이며 이번 12조건에서 같은 J를 얻었지만,
동일 J의 12조건 모두 full보다 case-median 시간이 길었습니다.
Static은 더 빠른 대신 일치 조건이 적습니다. 따라서 **조회 절감은 확인했지만 시간 우위는 없음**으로
해석합니다. query budget을 바꿔 full로 자동 전환하지는 않으며, 조회 제한이 없다면 이 작은 문제에서는
명시적 `--methods full`이 더 실용적인 선택이었습니다. 큰 지도에서 같은 우열을 보장하지 않습니다.

새 검산은 case/방법별 이산 최적값 60개(full 12개 포함), 선별 재생 48개, 반복 배치/clock 180개,
실제 조회한 후보의 fresh visibility 6,912,000개 항목과 독립 LOS 표본 816건을 통과했습니다.
독립 LOS 전수 증명이나 실제 RSU 환경 검증은 아닙니다. 상세 결과는
[보고서](results/geometry_placement_20261103/REPORT.md),
동일 J의 시간 비교와 단계별 시간은 [paired_latency](results/geometry_placement_20261103/paired_latency.json)를 봅니다.
기존 source 실험 결과는 보존했습니다. 결과는 ignored 경로입니다.

## Source 가림 응답 전이 (2026-10-23)

`connected_response_transfer_study.py`는 단독 점수 순위가 아니라 **source에서 관측한 가림
정보를 도로 상대 좌표로 전이해, 상대 센서와의 조건부 추가 기여 예측에 사용**하는 별도 경로입니다.
기존 직선 source의 광학적으로 가능한 ray에는 가림 label이 0건이었으므로, 그 자료만으로는
geometry 예측에 추가할 가림 정보가 없었습니다. 새 실행은 직선 1개와 좌/우 70도·85도 source
4개를 같은 공통 물리 모델에서 별도로 계산합니다. 기존 결과는 바꾸지 않습니다.

후보의 station/side/높이/상대 yaw/pitch를 대응시키고, target은 station/높이를 유지하면서
도로 폭으로 정규화한 횡좌표의 최근접 source 표본으로 대응합니다. 대응되지 않는 표본은
가림으로 간주하지 않고 target range/FOV 예측을 유지합니다. 다음 이진 surrogate를 사용합니다.

```text
B_source = optical_source AND NOT measured_source
predicted_target = optical_target AND NOT mapped(B_source)
```

- geometry-only, straight residual, nearest-source의 가림 없는 대조군, nearest-source residual 4종.
- 모든 군의 초기값/동률/fallback 순위는 **같은 target optical 단독 J**입니다.
- 달라지는 것은 미관측 pose의 예측 mask뿐입니다. 실제 측정한 상대 센서와 조건부 J를 계산합니다.
- source 선택은 폭/꺾임/곡선 길이의 고정 거리식이며 target coverage나 정답 배치를 보지 않습니다.
- 최종 조합/feasibility는 실제 조회 mask만 사용합니다. LS/MAPPO/학습은 이번 단계에 없습니다.
- 사전에 정의한 새 synthetic 형상 6개 x 기존 제약 4종, 조회 예산 1/2/4, 온라인 288개 + 기준선 24개.
  미사용 실도로 일반화, 큰 도로망 최적성, zero-shot 정책 전이를 입증하는 실험은 아닙니다.
- 전이한 가림의 false-visible/false-blocked도 온라인 결과를 고정한 **후에만** 전체 기준선과 비교합니다.
  잘못 전이한 가림은 유효 후보를 놓치게 할 수 있으며 개선 보장은 없습니다.

설계는 [configs/connected_response_transfer.json](configs/connected_response_transfer.json),
정의와 검증 범위는 [CONNECTED_DEPLOYMENT_PROTOCOL.md](CONNECTED_DEPLOYMENT_PROTOCOL.md)를 봅니다.

```bash
# 계획만 출력: 파일/학습/raycast 없음
.venv/bin/python connected_response_transfer_study.py

# 새 output 필요. source 구축 -> 온라인 고정 -> 전체 기준선 -> 독립 검산을 순서대로 실행
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_response_transfer_study.py \
  --output results/connected_response_transfer_20261023 --execute
```

진행/완료 여부는 [manifest](results/connected_response_transfer_20261023/manifest.json),
수치는 [summary](results/connected_response_transfer_20261023/summary.json),
[원자료](results/connected_response_transfer_20261023/comparison.csv),
검산은 [independent_validation](results/connected_response_transfer_20261023/independent_validation.json)으로 확인합니다.
**24/24조건 완료, 온라인 288개 + 기준선 24개 모두 feasible, 회귀 128 tests 통과.**
다음은 선언된 전체 이산 기준선 J와 일치한 조건 수입니다.

| 지점/side당 조회 예산 | Geometry-only | 꺾인 source 가림 전이 | 전이의 승/동률/패 |
|---|---:|---:|---:|
| 1 | 3/24 | 3/24 | 0/24/0 |
| 2 | 13/24 | 16/24 | 3/21/0 |
| 4 | 24/24 | 21/24 | 0/21/3 |

직선 residual과 가림 없는 source 대조군은 모든 예산에서 geometry-only와 같은 배치를 얻었습니다.
예산 2에서는 source 정보의 추가 효과가 관찰됐지만, 예산 4에서는 반대로 3조건 악화했습니다.
전이 방법 자체의 J는 예산이 늘어날수록 감소하지 않았습니다. 다른 방법과의 우열이 바뀐 것입니다.

전체 24조건의 eligible candidate-target 배열에서 실제로 가려졌는데 보인다고 예측한 항목은
116,320개에서 4,446개로 줄었지만, 실제 보이는데 가려졌다고 예측한 항목이 **40,998개** 생겼습니다.
표본/제약 profile이 반복되는 기술통계이며 검출 오탐률이 아닙니다. 단순 가림 전이는 일부 정보를
추가하지만 **geometry 차이에 따른 과도한 가림 전이와 negative transfer**를 해결하지 못했습니다.
시간 우위와 일관된 우월성은 입증하지 않았습니다.

검산은 전체 최적값 24개·후보군 최적값 288개·선별 재생 288개·배치 312개를 통과했고,
새 물리 계산 13,132,800개 항목 및 독립 LOS 표본 1,503건도 통과했습니다.
기존 prior 실험의 원본 2,273개 파일은 그대로입니다. 해석과 재현 명령은
[이번 보고서](results/connected_response_transfer_20261023/REPORT.md),
조건별 차이는 [paired_diagnostics](results/connected_response_transfer_20261023/paired_diagnostics.json)를 봅니다.
모든 결과는 ignored 경로입니다. 다음 검토는 관측된 source 전이 오류의 신뢰도/보정이며,
이를 현재 개발 결과에 맞춰 조정한 뒤 같은 맵에서 일반화됐다고 주장하지 않습니다.

## Source Prior 기여 분리 (2026-10-23)

`connected_prior_ablation.py`는 **동일한 순차 조건부 선별에서 사전 순위 정보만 변경**합니다.
이전 geometry 단독 대조군은 순차 탐색이 아니었으므로, 24/24 일치가 source 재사용 덕분이라고
분리해서 해석할 수 없었습니다. 이번에는 세 군 모두 동일한 조건부 선택과 전역 K=2 조합을 사용합니다.

| 사전정보 | 초기 후보·동률·대체 선택의 순위 |
|---|---|
| source | 기존 직선 source에서 실제 계산한 단독 센서 J |
| geometry | 새 지도에서 가림을 제외한 range/FOV 단독 J 예측 |
| random | seed=0,1,2,3,4의 고정 permutation, 모든 seed 보고 |

source 제거군은 초기값뿐 아니라 동률·대체 선택에서도 source 점수를 읽지 않습니다.
이 비교의 source 정보는 **현재 방법의 단독 J prior**이며, 모든 library 표현이나 학습된
response 전이 방법을 대표하지 않습니다. 실제 측정 mask와 공통 제약으로만 배치를 채택합니다.

- 지점/side당 실제 조회 예산 1·2·4개. 예산별 독립 cache이며, 선택 순서는 같은 prior의 prefix입니다.
- 하위 LS/MAPPO와 추가 pose 조회 없음. 후보군 내 feasible 배치가 없으면 실패로 저장하며 0점 처리하지 않습니다.
- 기존 형상 6개 × 제약 4종의 개발 자료. random 5회는 같은 지도에서의 반복이지 새 지도 5개가 아닙니다.
- 모든 온라인 배치를 먼저 고정한 뒤, 설치 가능한 전체 후보만 새로 조회해 이산 기준선을 계산합니다.
- 필요한 조건만 먼저 검사하는 opt-in 필터는 필터 없는 전체 검사와 대조했습니다. 기존 기본 선택 경로는 유지합니다.

시간은 scene/사전 순위/실제 probe/조합을 포함한 `decision_sec`와 공유 geometry 구성,
source 오프라인 구축, 최종 검산을 구분합니다. 프로세스 시작·source 로드·직렬화는 decision 시간에
포함되지 않으므로 전체 cold-start latency나 통제된 speedup으로 해석하지 않습니다.

**24/24조건 완료·검산 통과:** 온라인 결과 504개와 전체 기준선 24개 모두 제약을 만족했습니다.
다음은 전체 이산 기준선의 J와 일치한 수입니다. Random의 120회는 같은 24조건 × 5 seeds이며
독립 지도 120개가 아닙니다.

| 지점/side당 실제 조회 수 | Source | Geometry | Random 5 seeds |
|---|---:|---:|---:|
| 1 | 2/24 | 5/24 | 0/120 |
| 2 | 13/24 | 16/24 | 42/120 |
| 4 | 24/24 | 24/24 | 118/120 |

동일 조건의 source vs geometry J는 예산 1에서 **0승/8동률/16패**, 예산 2에서
**0승/16동률/8패**, 예산 4에서 **24동률**입니다. 평균 J 손실도 1개 예산에서
source 20.067 vs geometry 4.608, 2개에서 2.558 vs 0.383이었습니다. J 손실은 coverage %p가 아닙니다.
**현재 source 단독 점수 순위 재사용의 우위는 확인하지 못했습니다.** 기존 24/24 결과를
source library만의 기여로 해석하면 안 됩니다. 모든 library 재사용이 불가능하다는 결론도 아닙니다.

예산 4의 평균 decision 시간은 source 1.265초, geometry 1.278초, random 1.268초이며
설치 가능 후보만 사용하는 전체 기준선은 0.502초였습니다. 이번 작은 K=2 문제에서는 조회 수
절감이 시간 우위로 이어지지 않았습니다. 반복 cold-start 성능 검증과 구분합니다.

검증: **전체 기준선 24개·후보군 최적값 504개·선별 재생 504개·배치 528개**, 회귀 **119 tests** 통과.
source 예산 4의 선택/질의 순서 24개와 source/기준선 배치 48개도 이전 실행과 일치했습니다.
선별 재생은 동일 코드, ray 계수 검사는 공통 FOV 코드이므로 독립 LOS 물리 검증과 구분합니다.

```bash
# 계획만 확인, 파일/학습/raycast 없음
.venv/bin/python connected_prior_ablation.py

# 실제 비교. 기존 output이 있으면 새 이름 지정
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_prior_ablation.py \
  --output results/connected_prior_ablation_20261023 \
  --query-budgets 1 2 4 --random-seeds 0 1 2 3 4 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_connected_priors.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_connected_priors.py results/connected_prior_ablation_20261023
```

완료 여부와 수치는 [manifest](results/connected_prior_ablation_20261023/manifest.json),
[summary](results/connected_prior_ablation_20261023/summary.json),
[조건/seed별 CSV](results/connected_prior_ablation_20261023/comparison.csv),
[동일 조건 비교](results/connected_prior_ablation_20261023/paired.csv),
[독립 집계 검산](results/connected_prior_ablation_20261023/independent_validation.json)이 기준입니다.
[해석과 다음 연구 판단](results/connected_prior_ablation_20261023/INTERPRETATION.md)도 함께 봅니다.
결과는 ignored 경로로 clone에 포함되지 않습니다.

### 추가 검증 경로 (2026-10-23)

기존 검증기에 빠져 있던 **CSV의 지도/조건 식별자, source/input 파일 목록 완전성,
실행 계획의 수와 센서 설정, source 배열/후보 순위, 저장 mesh, 이산 최적성 표시와 후보 수,
기준선 ray/call 수와 시간 계측 경계**를 추가 대조합니다.
고의로 수치/식별자/mesh/표시를 바꾸거나 hash를 다시 쓴 축소 자료도 거부하는 테스트를 추가했습니다.
실험 알고리즘과 과거 결과는 수정하지 않았습니다. 보강 후 회귀 **122 tests**가 통과했습니다.

`connected_prior_verification.py`는 이 저장 배열 검산 뒤 source와 중복되지 않는 target
형상을 새 scene에서 다시 raycast합니다. 제약 profile 간 geometry/visibility가 같은지,
온라인 조회 mask와 최종 배치 mask가 새 계산과 같은지도 검사합니다.
도로 상대 좌표/자세와 range/FOV는 별도 수식으로, LOS는 층화 표본의 segment-triangle
교차 계산으로 대조합니다. **모든 LOS의 독립 전수 검증은 아닙니다.**
물리 검산 중 전체 grid를 조회하는 비용은 별도의 오프라인 검증 비용이며 온라인 실험에 공급하지 않습니다.

```bash
# --execute 없으면 계획만 출력, 파일/raycast 없음. 실제 검증에는 새 output 사용.
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_prior_verification.py \
  results/connected_prior_ablation_20261023 \
  --output results/connected_prior_verification_20261023 --execute
```

검증 산출물은 [별도 manifest](results/connected_prior_verification_20261023/manifest.json),
[저장 배열/최적값 검산](results/connected_prior_verification_20261023/strengthened_validation.json),
[물리 검산](results/connected_prior_verification_20261023/physical_validation.json),
[회귀 로그](results/connected_prior_verification_20261023/regression.log)에 저장합니다.
원본 폴더 아래에는 쓰지 않으며, 실행 전후 원본 파일 목록과 SHA-256을 대조합니다.
새 학습, 새 미사용 지도 성능 실험 또는 cold-start 시간 비교가 아닙니다.

**실제 재검증 완료:** 후보 pose 3,120개 / 관측 여부 7,084,800개가 새 계산과 같고,
독립 LOS 표본 920건도 일치했습니다. 배치 528개와 원본 파일 2,273개의 검사를 통과했습니다.
기존 수치 오류는 발견하지 못했으며 **source 단독 J prior 우위 없음**이라는 해석도 유지됩니다.
범위와 보완 사항은 [재검증 보고서](results/connected_prior_verification_20261023/REPORT.md)에 정리했습니다.

## 설치 배분과 자세 보정 분리 (2026-10-22)

검증에서 확인한 두 문제를 분리하기 위해 opt-in `connected_hierarchy_study.py`를 추가했습니다.
**상위 단계는 관측한 후보군에서 설치 지점/side/초기 자세를 공동 선택하고, 하위 단계는
선택한 설치 지점을 고정한 채 높이/yaw/pitch를 보정**합니다. 기존 8m station 이동 보정은
별도 대조군입니다. 아직 두 구간·K=2인 개발 파일럿이며 큰 도로망 전체 계층 최적화가 아닙니다.
기존 코드 경로, 설정과 과거 결과는 유지했습니다.

선별은 source 응답 재사용, target geometry의 가림 없는 range/FOV 예측, **이미 실제 관측한
상대 후보와의 조건부 J를 이용하는 순차 probe** 세 가지입니다. 모두 설치 지점/side당
4개 pose를 실제 조회하며, 하위 보정은 독립 cache에서 최대 16개 추가 pose만 조회합니다.
상위 설치 배분은 구간당 한 대 제한과 제한 해제를 비교합니다. 제한 해제는 탐색 공간 변경이지
학습 효과가 아닙니다. 반경 대조군의 구간 배분은 이동 후 달라질 수 있어 최종 배분도 저장합니다.

예측 mask로 feasible을 인증하지 않습니다. 상위 선택/하위 개선은 실제 관측 mask와 공통
제약을 사용하고 최종 배치는 fresh scene에서 검산합니다. 같은 round의 선택에는 이전 round까지
측정한 context만 허용합니다. 온라인 결과를 저장·hash 고정한 후에만 전체 후보 기준선을 만듭니다.
전체 target 좌표와 geometry는 알려져 있으며, 연결부 ±10m 바깥의 실제 관측을 자르지 않습니다.

**24/24조건·480개 배치 출력/검산 완료:** 이미 관찰한 형상 6개 × 제약 4종이며 독립 실도로
24개가 아닙니다. 다음은 보정 전 상위 선택의 결과입니다. 두 상위 배분 모드의 J는 모두 같았습니다.

| 후보 선별 | 전체 이산 기준선 J 일치 | 평균 J 차이 | 설치 지점 고정 보정 후 일치 |
|---|---:|---:|---:|
| 기존 source 응답 재사용 | 15/24 | 1.275 | 18/24 |
| Geometry 단독 예측 | 12/24 | 3.429 | 12/24 |
| 실제 관측 context 조건부 선별 | **24/24** | **0** | **24/24** |

J는 coverage 개수에서 중복·설치비용 패널티를 뺀 값입니다. coverage %나 %p gap이 아닙니다.
평균 J 차이는 `전체 이산 기준선 J - 방법 J`입니다.
조건부 선별은 source-only 대비 9개, geometry-only 대비 12개에서 J가 높고 나머지는 동률입니다.
**이번 자료에서 8m 이동을 허용해도 고정 지점 보정보다 더 좋아지지 않았습니다.** 구간 배분/하위
탐색의 일반적 효과나 source seed의 필요성을 입증한 것은 아닙니다.

설치 가능한 전체 자세 480개(금지 영역 적용 시 216개) 중 각각 160/72개를 선별 조회합니다.
검산용 전체 archive는 금지 후보까지 480개를 모두 생성했으므로 이 비용과 직접 나눠 speedup을
계산하지 않습니다. 기록된 평균 선별 시간도 source-only 0.090초, geometry-only 0.136초,
조건부 10.497초로 서로 다릅니다. 동일 시간 예산의 비교가 아니며 속도 우위는 확인하지 않았습니다.

조회 수가 같아도 ray 수·optical 계산량·실행시간은 다릅니다. 조건부 선별은 source를 seed/tie에
쓰는 수작업 규칙이고, 학습된 response 전이가 아닙니다. MAPPO 우위, speedup, 미사용 실도로
일반화나 전역 연속 최적성은 이 단계의 주장이 아닙니다. 수치 제약은 여전히 예시입니다.

```bash
# 계획만 확인: 파일/학습/raycast 없음
.venv/bin/python connected_hierarchy_study.py

# 실제 개발 비교: 새 output 필요
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_hierarchy_study.py \
  --output results/connected_hierarchy_20261022 --lower-extra-queries 16 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_connected_hierarchy.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_connected_hierarchy.py results/connected_hierarchy_20261022
```

정보/예산/계측 경계와 해석은 [CONNECTED_DEPLOYMENT_PROTOCOL.md](CONNECTED_DEPLOYMENT_PROTOCOL.md)의
`Hierarchy Redesign Pilot` 절을 확인합니다. 실제 완료 상태와 집계는 아래 결과 root의 manifest/summary가 기준입니다.
선별된 target 후보군/전체 후보의 최적값은 독립 전수조사하며, 선별/하위 정책 재생은 동일 구현을 사용하므로 독립 증명과 구분합니다.
결과는 ignored 경로로 clone에 포함되지 않습니다.
검증은 **전체 기준선 48개·선별 후보군 최적값 144개·선별 재생 72개·하위 재생 288개**와
공통/연결부 회귀 **111 tests**를 통과했습니다. 이전 배치/기준선 72개 재현 및 관측 기록 504개의
ray/call 계수도 확인했습니다. 후자는 공통 FOV 코드 재계산이지 독립 물리 검증은 아닙니다.

- [실행 상태](results/connected_hierarchy_20261022/manifest.json)
- [방법별 집계](results/connected_hierarchy_20261022/summary.json)
- [조건별 원자료](results/connected_hierarchy_20261022/comparison.csv)
- [검증 결과](results/connected_hierarchy_20261022/independent_validation.json)
- [결과 해석과 다음 검증](results/connected_hierarchy_20261022/INTERPRETATION.md)

## 확장 검증과 확인된 문제 (2026-10-22)

새 방법/학습을 추가하기 전에 `connected_validation_audit.py`로 최신 source-response
실험의 문제 정의·선별·전이·제약·LS·저장/집계를 점검했습니다. **원자료의 수치/좌표 오류는
발견하지 않았지만, 검산의 누락과 방법 설계의 한계를 확인했습니다.** 정식 학습·새 visibility
cache·대규모 solve 없이, 저장 배열과 독립적인 좌표/FOV/segment-triangle 검사를 사용했습니다.
작은 회귀 테스트의 임시 raycast/PPO 실행은 정식 실험과 구분합니다.

| 점검 | 확인 결과 | 해석 |
|---|---|---|
| 검산 누락 | LS 초기값과 실제 joint 결과의 연결, 선택 후보의 pool 포함, 세부 metric·CSV/paired/summary 검사를 보강 | 목적값이 같아도 잘못된 출처/집계를 통과시킬 수 있던 빈틈을 차단. 실제 저장 216개는 통과 |
| source 압축 근사 | 지점 20개 중 12개에서 greedy의 최악 source 응답 손실이 exact 압축보다 큼 | 근사 손실은 있으나, exact 압축으로 바꾼 저장-mask 반사실에서도 target J는 24개 모두 동일 |
| source 응답 목표 | 단독 최고 source 후보 고정을 해제하고 source-optimal 후보군 전체를 허용해도 조합 9/24는 target 최적 J 미달 | 현재 직선 source의 가상 context를 잘 보존하는 목표만으로 target 조건부 가치를 보장하지 못함 |
| 보정 도달 범위 | LS 72개를 재생해 배치·평가·조회 수가 일치. 모두 원래 측면/반경 영역의 최적 J | 이번 잔여 차이를 동일 행동공간의 학습량 증가만으로 없앨 수 없음 |
| 연결부 범위 | 다른 구간으로 넘어간 관측 target이 연결점에서 station 기준 28.5~34.5m까지 존재 | ±10m 서비스 영역은 상호작용 경계가 아님. 현 코드는 visibility를 그곳에서 자르지는 않음 |
| 제약의 실효성 | 높이·pole 용량·전체 중복 상한을 하나씩 제거해도 다른 제약 아래 feasible pair가 늘지 않음 | 설정된 제약 개수와 실제 난도는 다름. 비용/최소 coverage 등은 feasible 공간에 영향이 있으나 최적 J는 유지 |

source-optimal 후보군 분석은 **현재 source minimax/평균 손실 목표와 동일 4개 예산** 아래의
오프라인 oracle 진단입니다. 모든 library 설계가 실패한다는 증명도, 새 online 방법 결과도 아닙니다.
최적값을 바꾼 제약은 설치 금지 영역(12개 적용 중 5개)과 연결부 최소 중복(12개 중 11개)이었습니다.
비용·간격·고장 내성 등이 일반적으로 불필요하다는 뜻은 아닙니다. 수치는 여전히 예시 조건입니다.

독립 좌표 변환·range/FOV 검사 대상은 고유 형상 6개의 6,912,000개 candidate-target 항목이고,
별도 segment-triangle 교차 검사는 층화 표본 792개입니다. 저장 mesh의 전체 LOS를 독립 재계산하거나
실제 LiDAR 성능을 검증한 것이 아닙니다. 곡선의 균일 station/lateral 표본은 면적 균일 표본도 아닙니다.
24조건은 형상 6개 × 제약 4종의 개발 자료이며 독립 실도로 24개가 아닙니다.

```bash
# 계획만 출력, 파일/분석/학습 생성 없음
.venv/bin/python connected_validation_audit.py

# 이번 저장-mask 검증/반사실 분석. 기존 output 거부
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_validation_audit.py --execute

# 반복할 때는 새 output 지정
.venv/bin/python connected_validation_audit.py \
  --root results/connected_response_study_20261022 \
  --output results/connected_validation_repeat --execute
```

검증 root: `results/connected_response_study_20261022/validation_audit_20261022/`.
**원래 파일 650개 hash 불변, 새 테스트 11개와 연결부/공통 제약/RSU/pipeline 등 99개 통과**, 문법 검사와
기본 GREEDY/MILP/MAPPO dry-run 통과입니다. 기존 solver/선별/LS/physics 및 정식 결과는 바꾸지 않았습니다.

- [확인된 문제와 우선순위](results/connected_response_study_20261022/validation_audit_20261022/FINDINGS.md)
- [강화 검산 결과](results/connected_response_study_20261022/validation_audit_20261022/strengthened_validation.json)
- [제약별 실효성](results/connected_response_study_20261022/validation_audit_20261022/constraint_activity.csv)
- [source exact 압축과 target 반사실](results/connected_response_study_20261022/validation_audit_20261022/counterfactual.csv)
- [LS 도달 범위](results/connected_response_study_20261022/validation_audit_20261022/ls_domains.csv)
- [독립 물리 검사·구간 간 관측 범위](results/connected_response_study_20261022/validation_audit_20261022/physical_checks.csv)

## Source Coverage Response 후보 선별 (2026-10-22)

1단계 후속으로 **source 관측 mask의 조건부 추가 기여량을 보존하는 후보 선별**을
추가하고, 이미 관찰한 24개 개발 조건에서 비교했습니다. 지점/side당 후보는 세 방법
모두 4개이며, source 후보 240개 중 80개를 선별해 두 구간에 전이합니다.
새 target 정답을 보고 특정 pitch를 넣거나 후보 예산을 늘리지 않았습니다.

`src/methods/source_modes.py`의 새 규칙은 source의 빈 배치 또는 다른 지점 센서 하나를
context로 삼습니다. 단일 센서 J에 비용이 포함되며, source의 균일 중요도에서
`추가 J = 단일 J - (coverage 계수 + 중복 패널티) × 상대 센서와의 관측 교집합 수`입니다.
최고 단일 J로 시작해 context별 최고 응답을 놓치는 **최악 손실**을 가장 줄이는 후보를
순차 선택합니다. 동률은 평균 손실, 단일 J, ID 순입니다. 4개 부분집합의 전역 최적 압축을
푼 것은 아니며 source의 가상 context는 target의 연결 형상/제약을 인증하지 않습니다.

| 후보 선별 | 조합 직후 최적 J 일치 | 조합 직후 평균 J 차이 | LS 후 최적 J 일치 | LS 후 평균 J 차이 |
|---|---:|---:|---:|---:|
| source 점수 상위 4개 | 17/24 | 1.614583 | 17/24 | 1.614583 |
| 높이·방향 다양성 4개 | 15/24 | 1.450000 | 18/24 | 1.150000 |
| source response 4개 | 15/24 | 1.275000 | 18/24 | 1.150000 |

세 방식 모두 24/24 제약을 만족했습니다. 차이는 `전체 축소 이산 격자 최적 J - 방법 J`이며
coverage %p가 아닙니다. response 조합은 다양성 조합 대비 **3개 개선·21개 동률·0개 열세**였고,
개선은 폭 8m/±20도 중복 조건 두 개와 +20도 결합 조건 한 개에서 각 1.4 J입니다.
하지만 **LS 후에는 기존 다양성 방식과 24개 모두 J가 같고, 남은 6개 미달도 같습니다.**
동가 최종 배치가 가능하므로 pose 자체가 모두 같다는 뜻은 아닙니다.

결론은 제한적입니다. source response를 보는 구현은 완료했지만, 이 규칙만으로 기존
보정 후 성능을 넘지는 못했습니다. 학습 모델·전이 오차 보장·큰 도로망 해법의 완성이 아니며,
이 단순 규칙 자체를 노벨티로 주장하지 않습니다. 다음에는 연결 형상별 context와 설치선
배분의 표현을 검토하되, 측면/반경 행동 확장은 별도의 동일 행동공간 비교로 분리합니다.

```bash
# 계획만 출력; 기존 두 규칙/기본 CLI는 유지
.venv/bin/python connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json --response-pool

# 실제 실행. 기존 output은 거부하므로 반복 시 새 이름 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json --response-pool \
  --output results/connected_response_study_20261022 --execute

# 저장 배열/선별/목적값의 독립 검산; raycast/학습 없음
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  results/connected_response_study_20261022/source/tests/validate_connected_constraints.py \
  results/connected_response_study_20261022
```

**216개 저장 배치/직접 검산, 전체·구간별 기준값 48개, 후보 pool 최적값 72개**를 확인했습니다.
source 선별은 모든 target 질의 전에 고정했고, 독립 코드로 context·선별 순서·손실을 재현했습니다.
공통 설치 허용 후보 조회는 기본/중복 조건 160개, 설치 제한/결합 조건 72개로 세 규칙이 같습니다.
LS 추가 조회·ray 수·wall time은 별개이며 speedup 실험이 아닙니다. MAPPO는 실행하지 않았습니다.
새 테스트 7개와 기존 연결부/pipeline을 합친 47개 테스트, 문법 검사, 세 기본 dry-run이 통과했습니다.

- [결과 해석](results/connected_response_study_20261022/FINDINGS.md)
- [24개 조건 보고서](results/connected_response_study_20261022/REPORT.md)
- [216행 원자료](results/connected_response_study_20261022/comparison.csv)
- [선별 순서와 source context](results/connected_response_study_20261022/source_pools.json)
- [독립 검산](results/connected_response_study_20261022/independent_validation.json)

기존 실험 결과는 보존했습니다. 위 24개는 이미 관찰한 **개발 자료 재평가**로, 새 held-out
일반화 증거가 아닙니다. 생성 결과는 ignored 경로이므로 clone에 포함되지 않습니다.

## 후보 누락·보정 도달 범위 진단 (2026-10-22)

`connected_failure_diagnostics.py`로 위 연구의 첫 단계인 **실패 원인 분석**을 수행했습니다.
완료된 24개 조건의 저장 mask만 읽었고, 후보 규칙·원자료·실험 수치를 바꾸지 않았습니다.
두 후보 방식의 48개 조건별 진단을 저장했습니다. 새 정식 학습/raycast는 실행하지 않았습니다.

**핵심 결과: 남은 차이는 이번 탐색공간 안에서 LS가 더 좋은 해를 놓친 문제가 아니었습니다.**

| 후보 방식 | 조합 직후 최적 J 미달 | LS 후 미달 | 측면 고정 손실이 있는 조건 | 반경 손실이 추가되는 조건 | 현재 허용 공간 안의 탐색 손실 |
|---|---:|---:|---:|---:|---:|
| 상위점수 | 7/24 | 7/24 | 7 | 1 | 0/24 |
| 다양성 | 9/24 | 6/24 | 6 | 2 | 0/24 |

설치선 측면 고정과 반경 8m는 **보정 알고리즘의 탐색 제한**이며 물리적 설치 금지와 다릅니다.
손실은 `전체 후보 → 초기 측면 유지 → 초기 station 반경 유지 → 제약을 지키는 행동 연결 성분
→ 저장 LS` 순서로 분해했습니다. 위 조건 수는 서로 배타적인 분류가 아니며, 이 순서의
분해값을 유일한 인과 기여로 해석하지 않습니다.

다양성+LS의 6개 미달 조건은 다음과 같습니다. 단위는 coverage %p가 아니라 J입니다.

| 조건 | 전체 차이 | 측면 제한분 | 반경 제한분 |
|---|---:|---:|---:|
| 기본 / 폭 10m / -45도 | 0.80 | 0.80 | 0 |
| 기본 / 폭 10m / +45도 | 0.80 | 0.80 | 0 |
| 기본 / 폭 12m / -75도 | 10.80 | 5.65 | 5.15 |
| 기본 / 폭 12m / +75도 | 10.80 | 5.65 | 5.15 |
| 설치 제한 / 폭 10m / +45도 | 0.80 | 0.80 | 0 |
| 설치 제한 / 폭 12m / -75도 | 3.60 | 3.60 | 0 |

현재의 측면·반경 제한 아래 전체 후보를 다시 전수조사한 최적값이 LS와 48/48 일치했습니다.
따라서 같은 제한을 유지하는 한, 이번 차이를 학습 steps 증가만으로 없앨 수 없습니다.
이는 다른 지도에서 LS가 항상 좋거나 MAPPO가 불필요하다는 일반 결론이 아닙니다.

후보 압축의 문제도 대비됐습니다. 모든 동가 최적 조합을 살펴 누락 개수가 가장 적은
보충 witness를 골랐습니다. 상위점수 방식의 누락 기록 7건은 source 위치별 순위 9위인
`높이 5m / 상대 yaw ±45도 / pitch -20도`입니다. 다양성 방식은 실패 9조건의 누락 기록
12건 모두 `높이 3m / 상대 yaw 0도 / pitch -10도`이며 source 순위 2위였습니다.
서로 다른 조건에서 같은 source 후보가 반복되므로 12개 독립 후보/표본이라는 뜻은 아닙니다.

다양성 규칙이 높이·yaw별 최고 source pitch 하나만 남기며 유용한 다른 pitch를 버린
사례입니다. 선택한 최적 상대 센서를 고정하면 위 누락 후보의 target 위치별 조건부 순위는
모두 1위였습니다. **그 상대 센서는 정답을 본 뒤 고른 것이므로 온라인 선택 규칙이 아닙니다.**
동가 최적해에 다른 자세가 포함될 수 있어 witness 하나를 반드시 필요한 유일 pose로 보지 않습니다.

```bash
# 계획만 출력하며 파일/raycast/학습 없음
.venv/bin/python connected_failure_diagnostics.py

# 실제 저장 배열 분석. 반복할 때 새 --output 지정
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python connected_failure_diagnostics.py \
  --root results/connected_constraint_holdout_20261021 --execute

# 원래 실행의 frozen 전수조사 코드로 독립 대조; 새 검산 output 경로 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/validate_connected_diagnostics.py \
  results/connected_constraint_holdout_20261021/failure_diagnostics_20261022
```

- [진단 해석과 다음 검증](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/FINDINGS.md)
- [전체 진단 보고서](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/REPORT.md)
- [손실 분해 그림](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/loss_decomposition.png)
- [48개 조건별 진단](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/summary.csv)
- [누락 witness와 조건부 기여](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/omitted_candidates.csv)
- [독립 검산](results/connected_constraint_holdout_20261021/failure_diagnostics_20261022/independent_validation.json)

입력 파일 506개의 hash 불변, 제한 영역 최적값 96개, 모든 동가 최적 조합을 이용한 누락
판정 48개를 확인했습니다. 연결 성분 최적값은 `반경 영역 최적값 = feasible LS 값`이라는
상·하한 일치로 48개를 독립 확인했으며, 연결 성분의 상태 개수까지 별도 검산한 것은 아닙니다.
새 8개와 기존 연결 제약/전이/pipeline을 합친 테스트 25개 및 문법 검사가 통과했습니다.
회귀 테스트의 임시 소형 raycast/PPO 실행은 정식 결과 분석과 구분합니다.

다음은 pitch와 설치선 배분의 조건부 응답을 같은 후보 예산에서 보존하는 규칙을 설계하는
단계입니다. 이번 정답을 보고 후보를 추가한 값을 새 방법의 성능으로 쓰지 않으며,
규칙을 고친 뒤에는 별도의 미사용 geometry 평가가 필요합니다.

## 규칙 고정 후 새 폭·연결 형상 평가 (2026-10-21)

앞선 개발 실험에서 정한 source 후보 선택 규칙, 위치/side별 대안 4개, J와 제약값을
그대로 고정하고 **새 형상 6개 × 제약 4종 = 24개 조건**을 실행했습니다.
폭 8/10/12m, 연결각 각각 ±20/±45/±75도, 곡선 길이 각각 16/18/24m입니다.
각 도로의 폭은 일정하며 서로 다른 폭을 한 도로 안에서 연결한 실험은 아닙니다.

source는 **폭 7m·길이 40m인 직선 하나**입니다. target 폭별로 source를 다시 최적화하지
않았습니다. source의 station·설치선 측면·높이·상대 yaw·pitch는 보존하고, 횡방향
좌표만 새 설치선 `sign(d) × (target 폭/2 + 인도 폭/2)`로 옮깁니다.
source 후보 ID/배열·점수는 앞선 개발 실행과 같았고, target 질의 전에 해시로 고정했습니다.

**24/24 조건, 168개 방법 결과의 계산·독립 검산을 완료했습니다.**

| 방법 | 제약 충족 | 전체 축소 격자의 최적 J와 일치 |
|---|---:|---:|
| source 최고 배치 그대로 전이 | 12/24 | 0/24 |
| source 점수 상위 4개 조합 | 24/24 | 17/24 |
| 높이·방향 다양성 4개 조합 | 24/24 | 15/24 |
| 상위점수 조합 + LS | 24/24 | 17/24 |
| 다양성 조합 + LS | 24/24 | 18/24 |

다양성 조합은 상위점수 조합 대비 **7개 개선·11개 동률·6개 열세**였습니다.
개발 사례의 16/16 일치가 새 형상에서도 유지된 것은 아닙니다. 다양성 조합 뒤 LS가
3개 조건을 개선했지만, 여전히 6개 조건은 최적 J에 못 미쳤고 최대 차이는 10.8입니다.
이는 J 단위의 차이이며 coverage %p나 연속 공간의 최적성 gap이 아닙니다.

예를 들어 폭 12m/+75도/기본 조건에서 상위점수 조합은 coverage 98.785%, J=2834.85이고
다양성 조합은 coverage 99.688%, J=2824.05였습니다. 중복 관측 패널티를 포함하므로
coverage가 높다고 목적값도 높은 것은 아닙니다. 두 방법 모두 같은 J/제약으로 평가했습니다.

조합 단계의 실제 조회 후보는 기본/중복 조건에서 각각 160개, 설치 제한/결합 조건에서
각각 72개로 두 방법이 같았습니다. 전체 기준선 mask는 480개입니다. LS의 추가 조회,
서로 다른 제출 ray 수, 전처리·검산 비용을 포함한 동일 시간 예산이나 speedup은 아닙니다.
원래 전이의 제약 위반 12개도 coverage와 위반 사유를 남겼으며 성공 결과로 세지 않습니다.

```bash
# --execute 없으면 계획만 출력. 실제 반복은 새 output 경로 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --study configs/connected_constraint_holdout.json \
  --output results/connected_constraint_holdout_20261021 --execute

# 저장 배열 전수조사 검산. 새 raycast/학습/solve 없음
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_constraints.py \
  results/connected_constraint_holdout_20261021
```

- [전체 결과 보고서](results/connected_constraint_holdout_20261021/analysis/REPORT.md)
- [24개 조건 비교 그림](results/connected_constraint_holdout_20261021/analysis/overview.png)
- [75도 결합 조건 배치](results/connected_constraint_holdout_20261021/analysis/scene.png)
- [168개 방법 원자료](results/connected_constraint_holdout_20261021/comparison.csv)
- [독립 검산](results/connected_constraint_holdout_20261021/independent_validation.json)
- [고정 입력 설정](configs/connected_constraint_holdout.json)

worker는 2026-10-21 19:14:02~19:15:00 KST에 완료됐습니다. 저장 배치 168개 중 feasible은
156개이며, 전체/구간별 기준선 48개와 source 부분공간 최적값 48개를 독립 전수조사로
확인했습니다. 새 테스트 5개를 포함한 44개 테스트, 문법 검사 및 기존 세 방법 dry-run을
통과했습니다. source/input snapshot과 해시, 실행 의존성을 저장했습니다. 결과는 ignored 경로입니다.

이는 규칙 개발에 쓰지 않은 **합성 geometry 조합**의 확인이며 실도로 일반화나 각도 전 구간
보장은 아닙니다. 폭·각도·곡선 길이를 함께 바꿔 각 요인의 효과를 분리하지 못했습니다.
제약 수치는 여전히 예시이고, K=2·정적 mesh·유한 격자·표본 target에 한정됩니다.
MAPPO/검출/큰 도로망 최적화는 이번 실행에 없습니다. 이후 규칙을 바꾸면 이 자료는 개발셋입니다.

## 후보 다양성·연결부 중복·설치 제한 (2026-10-21)

`connected_constraint_study.py`에 높이·방향 대안 보존과 도로 상대 제약을 연결했습니다.
위치/side마다 source 점수 상위 4개를 남기는 방법과, 높이/상대 yaw의 서로 다른 mode를
4개 남기는 방법을 같은 후보 수로 비교합니다. mode별 pitch는 source J로 고르고,
최고 J에서 시작해 정규화된 높이·yaw의 최소 거리가 가장 큰 대안을 순차 추가합니다.
target 정답을 후보 선택에 사용하지 않습니다. 이 규칙은 앞선 개발 사례를 분석한 뒤 설계했습니다.

전체 80m·센서 2대·12m 곡선·±30/±60도에서 네 조건을 실행했습니다.

| 조건 | 연결부 요구 | 설치 제한 |
|---|---|---|
| 기본 | 추가 하한 없음 | 추가 금지 구간 없음 |
| 중복 | 연결부 coverage ≥95%, 이중 관측 target 비율 ≥50% | 없음 |
| 설치 제한 | 추가 하한 없음 | 양 끝·연결부 부근 및 첫 구간 오른쪽 설치선 금지 |
| 결합 | 연결부 coverage ≥95%, 이중 관측 target 비율 ≥50% | 위 설치 제한 동일 적용 |

연결부는 station 30~50m의 target입니다. 설치 금지는 양쪽의 [0,8], [34,46], [72,80]m와
오른쪽 설치선의 [0,40]m에 적용하며 경계를 포함합니다. left는 중심선 진행 방향 기준 +d,
right는 -d입니다. **수치는 기능·민감도 검증용 예시이며 현장 기준이 아닙니다.**
중복은 실제 검출·추적·handover 성능으로 해석하지 않습니다.

**16개 조건 × 7개 방법 = 112개 배치를 저장·검산했습니다.** 원래 전이 배치 8개는 설치
제약을 위반했습니다. 원자료에 infeasible과 위반 사유를 유지하며 성공 결과로 평균내지 않습니다.
두 후보 조합 방법은 16개 조건 모두 feasible 해를 찾았습니다.

| 후보 조합 | 전체 축소 격자의 최적 J와 일치 | 상대 비교 |
|---|---:|---|
| source 점수 상위 4개 | 13/16 | 비교 기준 |
| 높이·방향 다양성 4개 | 16/16 | 3개 개선, 13개 동률, 열세 0개 |

개선은 기본 조건 ±30도와 설치 제한 +30도에서 나타났고 J 차이는 3.7~4.15입니다.
두 방법의 조회 후보 수는 기본/중복 조건에서 각각 160개, 설치 제한/결합 조건에서 각각
72개로 같았습니다. 제출 ray 수와 wall time까지 같은 예산이라는 뜻은 아닙니다.
LS 추가 보정의 J 개선은 없었습니다. 이는 **이 개발용 4개 형상·고정 K·축소 격자**의 결과이며
미사용 실도로 일반화나 연속 공간 최적성, 도시 전체 배치 최적화를 입증하지 않습니다.

```bash
# 계획만 출력
.venv/bin/python connected_constraint_study.py

# 실제 실행; 반복할 때 새 output 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_constraint_study.py \
  --output results/connected_constraint_study_20261021 --execute

# 저장 mask에서 제약/전수조사/동일 조회 예산 독립 검산
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_constraints.py \
  results/connected_constraint_study_20261021

# 저장 결과만 분석하며 새 raycast/학습/solve 없음. 새 분석 output 필요
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python connected_constraint_report.py \
  --root results/connected_constraint_study_20261021 \
  --output results/connected_constraint_study_20261021/analysis_repeat
```

- [분석 보고서](results/connected_constraint_study_20261021/analysis/REPORT.md)
- [전체 조건 비교 그림](results/connected_constraint_study_20261021/analysis/overview.png)
- [금지 설치선과 60도 결합 조건 배치](results/connected_constraint_study_20261021/analysis/scene.png)
- [방법별 원자료](results/connected_constraint_study_20261021/comparison.csv)
- [독립 검산](results/connected_constraint_study_20261021/independent_validation.json)

설정은 [connected_constraint_study.json](configs/connected_constraint_study.json)입니다.
worker는 2026-10-21 17:58:04~17:58:38 KST에 완료됐습니다. 저장 배치 112개 중 feasible은
104개이며, 전체/구간별 기준선 32개와 library 부분공간 최적값 32개를 독립 전수조사로 확인했습니다.
새 테스트 5개와 기존 연결·곡률·제약·pipeline을 합친 39개 테스트, 문법 검사 및 기존 3개 방법
dry-run을 통과했습니다. source/input snapshot과 코드 hash를 저장하고 온라인 결과 후에만
전체 target mask를 생성합니다. 생성 결과는 ignored 경로이며 clone에는 포함되지 않습니다.
MAPPO는 이 실행에 없으며, 해당 학습을 연결하려면 같은 도로 상대 행동/제약 API가 필요합니다.

## 곡률·후보 대안 수 민감도 (2026-10-21)

`connected_curvature_sweep.py`로 전체 길이 80m·센서 2대·폭·센서 모델·격자·J·제약을
고정하고, ±30/±60도에서 곡선 구간을 20/15/12m로 바꿉니다. 같은 꺾임을 더 짧은
거리에서 만드는 12개 연결 형상입니다. 60도·12m 조건의 회전 반경은 약 11.46m입니다.

하나의 직선 source를 한 번 계산한 뒤, source만으로 정한 위치/side별 상위 대안
1/2/4개를 재사용합니다. target 후보 풀은 40/80/160개이며 전체 후보는 480개로 같습니다.
총 **36개 조건 × 6개 방법 = 216개 방법 실행을 완료**했습니다. source의 대안을 늘리는 효과와
실제 기하 변형의 효과를 구분하고, 같은 형상의 전체 후보 기준선이 풀 크기에 따라
변하지 않는지 독립 검산합니다.

```bash
.venv/bin/python connected_curvature_sweep.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_curvature_sweep.py \
  --output results/connected_curvature_sweep_20261021 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_curvature.py \
  results/connected_curvature_sweep_20261021
```

설정은 [connected_curvature_sweep.json](configs/connected_curvature_sweep.json),
실험 범위는 [통합 연구 프로토콜](CONNECTED_DEPLOYMENT_PROTOCOL.md)을 따릅니다.
실행은 source/input snapshot을 고정하고 새 output만 허용합니다. 시간은 반복 cold-start
비교가 아닙니다. 해당 곡률 실행에는 연결부 중복률 hard constraint·설치 금지 조건·MAPPO가
없으며, 이후 연결부 제약 실행은 위 별도 절을 기준으로 봅니다.

**이번 결과:** 곡선 구간을 20→15→12m로 줄여도 전이와 library 공동 선택의 coverage는
99.167%, 연결부 coverage는 100%였습니다. 후보 풀을 40→80→160개로 늘려도 공동 선택의
J는 같았고, 공동 선택 이후 LS 개선은 36개 조건 모두 없었습니다. 각 풀 크기에서 12개 형상
중 6개(±60도)는 전체 축소 격자의 최적 J와 같았고, ±30도의 차이는 J 3.7~3.8입니다.

±30도의 저장 기준선이 사용하는 5m 높이·±45도 상대 yaw 후보는 source 같은 위치에서
9위여서 상위 4개 풀에도 없습니다. 이번 결과는 단순 후보 수 증가보다 높이·방향·설치 측면의
다른 대안을 보존하는 규칙을 검토할 근거입니다. geometry 변화는 실제 visibility mask를
바꿨지만, 연결부 관측 실패나 재사용 가능한 최대 곡률을 찾지는 못했습니다.

- [분석 보고서](results/connected_curvature_sweep_20261021/analysis/REPORT.md)
- [곡률·후보 수 비교 그림](results/connected_curvature_sweep_20261021/analysis/coverage.png)
- [전체 방법 원자료](results/connected_curvature_sweep_20261021/comparison.csv)
- [독립 검산](results/connected_curvature_sweep_20261021/independent_validation.json): 배치 216개 및 기준선 72개 일치.

실험 worker는 2026-10-21 16:53:59~16:55:06 KST에 완료했습니다. 새 테스트 5개와 기존
연결/pipeline 검사를 합친 22개 테스트, 문법 검사와 기존 3개 방법 dry-run을 통과했습니다.
저장 배열만으로 source 순위와 visibility 변화 분석을 재생성하는 명령은 다음과 같습니다.
기존 분석 output을 거부하므로 재생성할 때는 새 이름을 사용합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python connected_curvature_report.py \
  --root results/connected_curvature_sweep_20261021 \
  --output results/connected_curvature_sweep_20261021/analysis_repeat
```

분석 폴더의 `provenance.json`은 보고서 코드와 입력 원자료 hash를 기록합니다.
원본 실행 snapshot/JSON은 변경하지 않으며, 생성 결과는 ignored 경로여서 clone에 포함되지 않습니다.

## 연결 geometry와 배치 후보 조합 (2026-10-20)

`connected_corridor_study.py`는 40m 직선 source에서 위치별 자세·높이 대안을 저장하고,
40m + 40m 연결 도로로 전이합니다. 중앙 20m 원호의 방향 변화는
0 / ±15 / ±30 / ±60도입니다. 광선은 곡선을 따라 휘지 않고 실제 3D mesh에서 직선으로 계산합니다.
양 끝은 유한한 실제 도로 끝이며, 이 실행은 원통형 주기 경계를 사용하지 않습니다.

- 비교: 독립 최선 배치 복제, 전이+LS, library 공동 선택, 공동 선택+LS, 두 전수조사 기준선.
- 모든 방법은 **실제 센서 2대**, 공통 J와 비용·간격·높이·coverage·단일 고장 coverage 제약.
- source 후보 240개에서 위치/side별 20개 대안을 유지해 target 후보 40개로 전이합니다.
- 전체 target 후보는 480개이며, 전역 기준선은 114,960개 쌍을 모두 고려합니다.
  별도 구간당 1대 기준선은 57,600개 쌍으로, 후보 축소와 센서 배분 제한을 구분합니다.
- 온라인 결과를 저장한 뒤에만 전체 후보 기준선을 계산합니다. 각 방법의 target memo는 독립입니다.
- 새 좌표의 LS는 station·높이·상대 yaw·pitch 이웃 탐색이며 기존 XY 11-action/MAPPO와 다릅니다.
- 전수조사 최적성은 **이 축소 격자·고정 K·정적 mesh·표본 target·제약 안에서만** 성립합니다.
  몇 각도에서 잘됐다고 그 사이 모든 곡률이나 실도로에 대한 성능을 보장하지 않습니다.

```bash
# 계획만 확인
.venv/bin/python connected_corridor_study.py

# 실제 제한된 파일럿; 기존 output은 거부하므로 반복에는 새 이름 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u connected_corridor_study.py \
  --output results/connected_corridor_pilot_20261020 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_connected_corridor.py
```

구현·제약·기하와 시간 계측의 정확한 범위는
[연결 배치 연구 프로토콜](CONNECTED_DEPLOYMENT_PROTOCOL.md)을 기준으로 봅니다.
예시 제약은 운영 기준이 아닙니다. 기존 LS/MAPPO 결과와 새 격자의 수치를 합치지 않습니다.

**완료 결과:** 위 명령으로 7개 각도 × 6개 방법 = **42개 배치**를 저장했고 모두 제약과
fresh-scene raycast 검산을 통과했습니다. 후보 40개 공동 선택은 7개 중 5개에서 전체 480개
후보의 최적 J와 같았습니다. ±30도에서는 공동 선택 coverage 99.167%, 기준선 99.583%,
J 차이 3.8입니다. 공동 선택 후 LS의 추가 개선은 없었습니다.

독립 전이도 모든 각도에서 coverage 99.167%였고 연결부 coverage는 모든 방법에서 100%입니다.
따라서 이번 개선은 주로 중복 관측 감소이며, 연결부 관측 실패나 재사용 가능한 최대 각도를
찾은 실험은 아닙니다. ±30도의 저장 기준선은 선택된 배치와 반대 설치 측면 및 누락된
높이/자세 후보를 사용하므로, 다음에는 학습량보다 대안 배분과 탐색 가능 범위를 먼저 점검합니다.
원통형 반복 경계와 MAPPO는 이 실행에 없습니다.

- [실행 보고서](results/connected_corridor_pilot_20261020/REPORT.md)
- [42행 원자료](results/connected_corridor_pilot_20261020/comparison.csv)
- [60도 연결 도로와 배치 비교](results/connected_corridor_pilot_20261020/angle_06/comparison.png)

아래 별도 검산은 저장 mask에서 모든 쌍의 제약·목적값을 독립 계산하여 14개 기준선과
42개 저장 배치를 확인합니다. 새 raycast·학습·파일 수정 없이 검산기/manifest hash를 출력합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_connected_corridor.py \
  results/connected_corridor_pilot_20261020
```

새 테스트 10개를 포함한 회귀 54개 중 53개 통과, 선택 실행 항목 1개 건너뜀입니다.
생성 결과는 ignored 경로이며 clone에 포함되지 않습니다.

## Coverage + 비용·제약 보정 경로 (2026-10-20)

검출·추적은 제외하고 **기하 coverage와 설치 조건**을 함께 평가하는 별도 opt-in 경로를
추가했습니다. 기존 순수 coverage 실험·기본 YAML·결과 파일은 변경하지 않습니다.
이 단계는 큰 도로망 전체 최적화나 MAPPO 우위의 입증이 아니라, 같은 조건으로 비교하기 위한 기반입니다.

- 공통 평가: union coverage, 중요 영역 가중 coverage, 초과 관측 penalty, 센서 수·설치비용 penalty.
- hard constraints: K 상한, 비용 예산, 최소 XY 간격, 같은 XY 지주당 센서 수, 설치 금지 직사각형,
  전체/지정 영역의 최소 coverage 및 최소·최대 이중 관측 비율.
- LS와 MAPPO 모두 **동일한 목적값 J와 hard constraints**로 최고 feasible 배치를 보존합니다.
  새 MAPPO는 기존 거리/unique/outward reward shaping을 사용하지 않습니다.
- 기본 경로는 초기 전이 배치가 제약을 어기면 `initial_infeasible`과 위반 항목을 저장합니다.
  `--repair-initial`을 명시하면 아래 공통 초기값 복구 단계를 먼저 수행하며 제약은 완화하지 않습니다.
  MAPPO의 위반 joint action은 전체 취소하고 -1 보상을 주며,
  샘플링한 action/log-probability를 그대로 PPO에 기록합니다. 허용 행동을 다시 뽑는 방식이 아닙니다.
  유효 이동의 raw reward는 `J(next)-J(current)`이며 PPO buffer의 scaling/clipping은 source 설정을 유지합니다.

목적함수는 기존 공통 엔진과 같은 target 개수 단위이며 자동 정규화하지 않습니다.
`d_j`는 target j를 관측하는 센서 수, `a_j`는 중요도, `c(p_i)`는 설치비용입니다.

```text
J(P) = w_cov * sum_j 1[d_j >= 1]
     + w_importance * sum_j a_j * 1[d_j >= 1]
     - w_overlap * sum_j max(d_j - 1, 0)
     - w_count * |P| - w_cost * sum_i c(p_i)
```

중복 제약의 `overlap_ratio`는 `count(d_j >= 2) / N`이며 위 penalty의 초과 관측 횟수와 다릅니다.
영역별 비율은 해당 직사각형 안의 모든 높이 target을 분모로 계산합니다.
중요도는 기본 1이고 영역이 겹치면 지정 factor 중 최대값을 사용합니다. 비어 있는 target 영역은 오류입니다.
설치비용은 `base_per_sensor + per_height_m * z + 해당 영역 surcharge 합`입니다.
z는 절대 설치 높이이고, 비용은 센서별 합산이며 공용 지주 공사비를 나눠 내는 모델이 아닙니다.

설정: [coverage_constraints_example.json](configs/coverage_constraints_example.json).
**예산·중복률·최소 coverage·비용 계수는 동작 확인용 임의 예시이며 현장 요구 조건이 아닙니다.**
기존 개발 target 420002/K3에 맞는 영역을 명시했으며 다른 지도에서는 영역을 따로 정의해야 합니다.
중복 확보가 실제 검출·추적 연속성이나 handover를 보장한다는 뜻은 아닙니다.

```bash
# 계획만 출력: 파일 저장, raycast, 학습, solver 실행 없음
.venv/bin/python prototype_constrained.py

# 짧은 통합 점검: 기존 output은 거부하므로 재실행에는 새 root 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python prototype_constrained.py \
  --output results/constrained_coverage_smoke_v2_20261020 \
  --steps 8 --evaluations 15 --seconds 30 --radius-m 2 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_constrained_coverage.py
```

runner는 source geometry/pose/config만 고정해 기존 구간 비율 전이를 재생성하고,
LS와 MAPPO를 fresh CPU process에서 순차 실행합니다. 두 방법은 visibility memo를 공유하지 않습니다.
`--initialization transfer_low|transfer_high|interpolation`을 명시하며 target coverage로 초기값을 고르지 않습니다.
원본 source는 아직 `best_known`일 수 있습니다. target 전체 후보 cache, 과거 target 정답,
사전학습 정책을 보정에 공급하지 않습니다. library 결과는 ignored 경로이므로 clone에 포함되지 않습니다.

결과 root에는 source/input snapshot·hash, `manifest.json`, `requirements.json`, `settings.json`, `packages.txt`,
worker log, `summary.csv`, 방법별 `result.json`, 초기/배치 NPY 및 density NPZ가 있습니다.
MAPPO는 정책·학습 로그·final-policy 배치를 별도로 저장합니다.
최종 배치 선택은 J 기준이므로 coverage 자체의 비감소를 보장하지 않습니다.
초기/선택/최종 정책 배치의 density·J·제약은 별도 direct pose raycasting으로 검산합니다.
`optimization_sec`과 `validation_and_save_sec`, import 등을 포함한 `worker_wall_sec`를 구분하고,
`runtime`의 ray 계수는 검산까지 포함하므로 기존 온라인 시간과 바로 나누지 않습니다.

최종 통합 점검 `results/constrained_coverage_smoke_v2_20261020/`은 두 방법 완료,
동일 초기 pose/requirements hash, 최종 제약 및 direct 검산 통과를 확인했습니다.
MAPPO는 8 steps/1 PPO update, LS는 15 이웃 평가뿐이므로 성능 우열·학습 수렴 근거가 아닙니다.
새 테스트는 공통 점수 일치, 비용/영역/지주 제약, 위반 행동 거부, 평가 상태 복원,
J 증가와 coverage 감소가 공존하는 사례, 8-step PPO, deadline과 dry-run 보호를 검사합니다.
아래 경량 묶음은 42개 중 41개 통과, 전체 후보 parity opt-in 1개 skip입니다.
문법 검사와 기존 MILP/GREEDY/MAPPO dry-run 세 가지도 통과했습니다.

```bash
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_constrained_coverage test_common_placement \
  test_pipeline test_local_mappo test_prototype_runtime test_placement_ui
```

이 v1 경로는 **전이된 센서 수를 고정**하므로 제거/추가 action이나 최소 K 결정은 없습니다.
따라서 count penalty는 같은 K의 보정 중 상수입니다. 복구 후 LS/MAPPO는 feasibility를 유지하는 이동만 허용합니다.
아래 v2에는 단일 센서 고장과 hub 거리/용량 검사를 추가했지만, 이동 경로의 연속 충돌·실제 전원/통신
경로와 QoS까지 모델링한 것은 아닙니다.
이 목적/제약에 대한 새 MILP 기준선, 충분한 seed·quality-time 비교, 실제 설치 요구값 확정은 후속입니다.
기존 순수 coverage MILP gap을 이 문제의 optimality 인증으로 재사용하지 않습니다.

### 공통 초기값 복구

운영 예산·필요 중복률은 아직 미정입니다. 예시값을 운영 기준으로 확정하거나 검출 성능 조건으로
해석하지 않고, **제약을 위반한 전이 초기값에서 시작하는 경우**를 처리하는 기능을 추가했습니다.
이 기능은 기본 비활성이고 `--repair-initial`로만 실행합니다.

```text
Source geometry/pose -> 원래 전이 배치 P0 (anchor)
  -> 공통 feasibility repair 1회 (정책 학습 없음)
  -> feasible.npy 저장, pose/요구조건/file hash 고정
  -> LS와 MAPPO가 동일한 복구 배치에서 각각 fresh process로 보정
```

복구는 기존 11-action 중 한 센서씩 이동하는 결정적 국소 탐색입니다.
공통 evaluator의 `constraint_residuals` 합계를 먼저 줄이고 동률이면 J가 높은 쪽을 고릅니다.
비용 초과는 `max(총 비용-예산,0)/max(예산,1)`, coverage 부족은 %p/100,
중복률은 부족/초과 비율, 거리 부족은 최소 간격으로 정규화한 센서 쌍 평균입니다.
센서 수·중복 pose·지주 용량·금지 영역은 해당 위반 수를 센서 수 또는 K로 정규화합니다.
이는 제약 간 절충을 위한 **휴리스틱**이며, 모든 개별 위반이 매번 감소한다는 보장은 없습니다.
복구 과정은 J나 coverage를 낮출 수 있으며, 이후 보정의 비감소 기준은 복구 후 feasible 배치입니다.
연속적인 임의 pose, 지도 밖 위치를 복구하는 투영기가 아니라 물리 격자 위 전이 배치의 추가 제약을 다룹니다.

반경은 복구 후 위치로 재설정하지 않고 **센서별 원래 전이 위치 P0**를 기준으로 유지합니다.
복구에 2m를 이동한 뒤 보정에 추가로 2m를 허용해 총 4m로 늘리는 방식이 아닙니다.
복구 단계의 raycast memo는 두 보정 worker에 전달하지 않습니다.
원래 배치가 feasible이면 변경 없이 저장하고, 복구 중 센서 수나 제약값은 바꾸지 않습니다.

| 복구 상태 | 의미 |
|---|---|
| `already_feasible` | 원래 전이 배치가 제약 만족, 복구 이동 없음 |
| `repaired` | 지정 제약과 원래 anchor 반경 안에서 feasible 배치 발견 |
| `no_feasible_found` | 국소 정체/시간/평가 한도 안에 찾지 못함. 불가능 증명 아님 |
| `infeasible_fixed_count` | 아래 단순 필요조건 위반으로 현재 고정 센서 수/영역에서 불가능 확인 |

필요조건 검사는 고정 센서 수의 K 초과, 최소 설치 높이에서의 기본 비용 하한,
센서 1대로 양의 이중 관측 요구, 두 anchor 거리+2r보다 큰 최소 간격 요구만 다룹니다.
비용 하한과 거리 상한은 공통 pose 검사의 수치 허용오차까지 보수적으로 반영합니다.
이는 센서 수를 변경하거나 반경을 넓힌 문제, 도시 전체 at-most-K 문제의 infeasibility 인증이 아닙니다.
복구 실패 시 LS/MAPPO를 실행하지 않고 root 상태는 `no_feasible_initial`, 종료코드는 2입니다.
이를 coverage=0인 실행으로 집계하지 않습니다.

동작 확인용 [복구 예시 설정](configs/coverage_constraints_repair_example.json)은 기존 예시에
작은 설치 금지 영역을 더해 일부 전이 pose가 위반하도록 만든 **통제된 기능 테스트**입니다.
현장 설치 제한이나 holdout 성능 평가가 아닙니다. 기존 예시 설정은 보존했습니다.

```bash
# 계획 확인만: 복구/학습/파일 저장 없음
.venv/bin/python prototype_constrained.py --repair-initial \
  --requirements configs/coverage_constraints_repair_example.json

# 실행 완료한 짧은 통합 점검. 재실행에는 새 output 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python prototype_constrained.py \
  --requirements configs/coverage_constraints_repair_example.json --repair-initial \
  --repair-evaluations 100 --repair-seconds 30 \
  --steps 8 --evaluations 15 --seconds 30 --radius-m 2 \
  --output results/constrained_coverage_repair_smoke_v2_20261020 --execute
```

이 점검에서 복구 21회 이웃 평가 후 센서 한 대가 2m 이동해 금지 영역을 벗어났습니다.
LS 15 이웃 평가와 MAPPO 8 steps/1 PPO update가 완료됐으며, 같은 복구 초기 pose hash,
원래 anchor 유지, 초기/최종 배치의 direct raycast 및 제약 검산을 확인했습니다.
학습 수렴이나 방법 간 우위의 근거로 사용하지 않습니다.

`repair/`에는 원래 `initial.npy`, `anchors.npy`, 탐색 `best_candidate.npy`, density NPZ,
위반량 trace와 필요조건 근거를 담은 `result.json`이 있습니다. `feasible.npy`는 성공할 때만 생성합니다.
실패한 `best_candidate.npy`는 배포 가능한 해가 아닙니다. 두 보정 폴더의 `initial.npy`는 복구 후 배치,
`anchors.npy`는 원래 전이 배치이며 MAPPO checkpoint에도 두 좌표를 구분해 저장합니다.
초기/후보의 direct raycast 검증은 위반 여부도 그대로 비교합니다.

복구의 시간·ray 수는 `repair/result.json`, 전체 복구 worker 시간은 root manifest의
`initial_repair.worker_wall_sec`로 별도 기록합니다. `summary.csv`의 `repair_worker_wall_sec`는
두 방법이 참조하는 **한 번의 공통 비용**이므로 비교 행마다 반복된 값을 합산하지 않습니다.
기존 `optimization_sec`에 복구 비용을 숨기지 않으며 통제된 speedup은 아직 보고하지 않습니다.

```bash
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_feasibility_repair test_constrained_coverage \
  test_common_placement test_pipeline test_local_mappo test_prototype_runtime test_placement_ui
```

최신 회귀는 53개 중 52개 통과, 전체 후보 parity opt-in 1개 skip입니다. 문법 검사와
기존 MILP/GREEDY/MAPPO dry-run도 통과했습니다. 새 복구 기능 자체는 solver 라이선스를 요구하지 않습니다.
`--repair-initial` 없는 기존 경로의 짧은 LS 실행도 별도 새 output에서 통과했습니다.
추가로 같은 복구 예시를 `--repair-seconds 1e-9`로 실행한
`results/constrained_coverage_repair_deadline_20261020/`에서 0회 이웃 평가, `no_feasible_found`,
LS/MAPPO `not_run`, 종료코드 2를 확인했습니다. 불가능 증명이나 학습 실행으로 집계하지 않습니다.

운영 요구값 확정, 더 넓은 feasible 탐색, 새 제약용 MILP 기준선,
반복 quality-time 비교는 여전히 후속 작업입니다.

### RSU 근거 기반 제약과 학습량 확장

128 steps/1 PPO update는 연결·민감도 파일럿이며 MAPPO 성능을 판단할 학습량이 아닙니다.
후속 구현과 문헌 근거·미검증 항목은 [RSU_COVERAGE_PROTOCOL.md](RSU_COVERAGE_PROTOCOL.md)에 정리했습니다.
검출/추적은 제외하고 비용·설치 영역·ROI coverage/중복률에 **장착 높이 제한,
전원/데이터 서비스 지점의 거리·포트 용량, 임의 센서 한 대 고장 후 최악 coverage**를 추가했습니다.
`constrained_coverage_v2`로 명시적으로 활성화하며 기존 v1/기본 CLI/과거 결과는 보존합니다.
LS와 MAPPO 모두 같은 목적함수·제약과 공통 복구 초기값을 사용합니다.

서비스 지점 연결은 수평 거리와 capacity의 이분 매칭 검사이지 실제 케이블 경로/전압/통신 QoS 보장이 아닙니다.
N-1 coverage는 센서별 mask에서 직접 계산하고 독립 direct raycast mask를 저장합니다.
문헌은 각 요인의 필요성을 뒷받침하지만 예시 35m/포트 2개/높이 3~5m/coverage 임계값의 운영 근거는 아닙니다.
아직 실제 사이트 요구값이 없으므로 모두 **실험용 수치**로 표시합니다.

`rsu_coverage_study.py`의 기본 계획은 합성 개발 지도 8개(T자/십자) × K3,4 × 제약 profile 3개 ×
seed 5개 × 4,096/20,000/100,000 steps = 720 paired jobs입니다. **전체 격자는 실행하지 않았습니다.**
초기 정책은 매 budget마다 새로 만들며, 실제 step/PPO update와 시간 제한에 의한 미달을 별도 기록합니다.
학습량을 늘리는 것 자체가 수렴이나 MAPPO 우위를 뜻하지 않습니다. 곡선/사교차로/실도로 검증은 별도 과제입니다.

```bash
# 기본은 출력만 하는 dry-run
.venv/bin/python rsu_coverage_study.py

# 두 개발 지도에서 긴 연결 점검. 재실행은 새 output 사용.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -u rsu_coverage_study.py \
  --target-ids 420002 420004 --k-values 4 --profiles resilient \
  --seeds 0 --step-budgets 4096 \
  --output results/rsu_coverage_development_20261020 --execute
```

실행 상태와 수치는 [manifest](results/rsu_coverage_development_20261020/manifest.json),
완료 후 [REPORT.md](results/rsu_coverage_development_20261020/REPORT.md)와
[raw_methods.csv](results/rsu_coverage_development_20261020/raw_methods.csv)를 기준으로 확인합니다.
이 경로 역시 개발 실험이며 20,000/100,000-step 다중 seed 정식 비교나 실환경 성능 증거는 아닙니다.

**2026-10-20 두 지도 점검 완료:** LS 2개와 MAPPO 2개 모두 저장·검산을 통과했습니다.
MAPPO는 각각 실제 4,096 joint steps/16 PPO updates를 수행했으며 학습량 미달은 없습니다.
T자 coverage는 LS 99.903%, MAPPO 99.673%; 십자는 LS 99.836%, MAPPO 98.990%였습니다.
한 대 고장 후 최악 coverage는 T자 67.603%/67.833%, 십자 75.873%/76.258%(LS/MAPPO)입니다.
MAPPO의 이 지표는 조금 높지만 선택 기준인 공통 J는 두 지도 모두 LS보다 낮았습니다.
단일 seed/K4 개발 점검이며 수렴·일반화·방법 우열을 확정하는 결과가 아닙니다.
문법 검사와 관련 회귀 71개 중 70개 통과/선택적 전체 후보 parity 1개 skip을 확인했습니다.

같은 원자료에서 J와 시간까지 표시하면 다음과 같습니다. 시간은 공통 복구·process 시작·최종 검산을
제외한 `optimization_sec`이며 동일 시간 예산의 speedup으로 해석하지 않습니다.

| target / 방법 | Coverage % | J | N-1 최악 coverage % | 보정 시간, 초 |
|---|---:|---:|---:|---:|
| 420002 T자 / LS | 99.903 | 31353.75 | 67.603 | 9.07 |
| 420002 T자 / MAPPO | 99.673 | 31085.75 | 67.833 | 109.83 |
| 420004 십자 / LS | 99.836 | 39756.10 | 75.873 | 9.57 |
| 420004 십자 / MAPPO | 98.990 | 39264.75 | 76.258 | 124.47 |

검출을 넣지 않아도 제약부 비교는 가능하지만, 제약이 많다는 사실만으로 RL 우위를 뒷받침하지 않습니다.
720 paired jobs 전체나 20,000/100,000-step 다중 seed 결과로 위 네 행을 설명하지 않습니다.

### 예시 제약 민감도 비교

`constrained_sensitivity.py`는 고정된 개발 지도 420002/K3의 low 전이에서 제약을 하나씩 바꿉니다.
운영 요구값이 미정이므로 [조건 명세](configs/constrained_sensitivity.json)는 **예시 기반 개발 파일럿**입니다.
기존 실험과 기본 설정은 보존하며 조건을 결과에 맞춰 자동 조정하지 않습니다.

| 조건 ID | 기준 조건에서 변경하는 항목 |
|---|---|
| baseline | 기존 coverage_constraints_example.json 그대로 |
| cost_38 / cost_39 | 총 비용 상한 60 → 38 / 39 |
| spacing_24 / spacing_36 | 최소 XY 간격 3m → 24m / 36m |
| overlap_025 / overlap_075 | junction 영역 최소 이중 관측률 0.02 → 0.25 / 0.75 |
| forbidden_1 / forbidden_6 | (24m,42m) 중심, 반폭 1m / 6m의 설치 금지 정사각형 추가 |

원래 전이 배치와 지도/target, 목적함수 가중치·비용 모델은 모두 고정합니다.
금지 영역은 앞선 복구 기능 테스트를 확장한 통제된 예시이고 실제 공사·설치 제한이 아닙니다.
9개 조건 × seed 0/1/2 = 27개 paired 반복이며, 각 반복은 공통 복구 1회 후 가능한 경우 LS/MAPPO를 실행합니다.
LS의 seed별 실행은 새 기하 표본이 아니라 timing 반복입니다. 복구도 같은 입력의 결정적 탐색 반복입니다.
실제 method 완료/차단 수는 root manifest와 집계표로 확인하며 차단을 coverage=0으로 처리하지 않습니다.

파일럿 설정은 보정당 6초 soft limit, MAPPO 최대 128 joint steps, LS 최대 128 이웃 평가,
복구 최대 128 이웃 평가/6초, 원래 전이 위치 기준 반경 4m입니다.
초기화·잔여 PPO·최종 정책 평가 때문에 실제 시간은 상한을 넘을 수 있으며 실제 step/update 수와 초과분을 저장합니다.
단일 seed의 더 긴 실행보다 반복과 조건 진단을 우선하는 짧은 파일럿이며 정책 수렴 실험이 아닙니다.

```bash
# 계획만 출력: 파일 생성/복구/학습/solver 없음
.venv/bin/python constrained_sensitivity.py

# 전체 예시 비교. 기존 output은 거부하므로 반복 실행에는 새 이름 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python constrained_sensitivity.py --execute

# 저장 원자료 재검사와 파생 보고서만 재생성
.venv/bin/python constrained_sensitivity.py \
  --output results/constrained_sensitivity_pilot_20261020 --report-only
```

상태: [manifest.json](results/constrained_sensitivity_pilot_20261020/manifest.json).
보고서: [REPORT.md](results/constrained_sensitivity_pilot_20261020/REPORT.md),
[repair_summary.csv](results/constrained_sensitivity_pilot_20261020/repair_summary.csv),
[summary.csv](results/constrained_sensitivity_pilot_20261020/summary.csv).
그림: [sensitivity.png](results/constrained_sensitivity_pilot_20261020/sensitivity.png).
실행 중에는 완료된 반복만 집계하며 대기 건을 성능 0으로 채우지 않습니다.

- `frozen/`: 실행 전 고정한 코드·source pose/config/geometry·조건별 요구값과 hash. target 전체 cache/정답은 없음.
- `runs/<condition>_seed<N>/`: 반복별 공통 복구, LS/MAPPO 원자료와 별도 direct 검산 기록.
- `raw_repairs.csv` / `raw_methods.csv`: 성공·필요조건 위반·미발견·method 차단, 실제 시간/학습량/ray 수와 영역 지표.
- `summary.csv`: 완료 method만의 평균·표본 SD(ddof=1). 완료가 1개뿐이면 SD는 공란이며, 차단도 공란입니다.
- `paired.csv`: 같은 복구 초기값/요구값의 MAPPO−LS 목적값·coverage 차이. 다른 초기값이면 집계 실패.
- `quality_time.csv`: 실제 보정 경과 2/4/6초까지 기록된 최고 J. 늦게 발견한 final 개선을 앞당겨 넣지 않음.
- `quality_summary.csv` / `paired_quality_time.csv`: 시각별 관측 수·평균/표본 SD와 두 방법 모두 관측된 반복만의 차이.
- `validation_summary.json`: 원자료 hash, 동일 source/target/요구값/원래 anchor·저장 density 재계산 확인 범위.

최종 best와 정시 quality-time을 구분합니다. 시간별 trace는 **보정 초기화 시작 기준**이며 transfer,
공통 복구, process import, 독립 검산 비용은 포함하지 않습니다. 별도 worker wall time과 함께 읽어야 합니다.
그 시각에 최초 관측조차 없으면 공란이고, 일찍 종료한 방법의 이미 얻은 결과는 이후에도 유지합니다.
trace 모든 중간 배치를 독립 raycast로 재계산한 것은 아니며 초기/최종 배치는 worker에서 direct 검산했습니다.
이 집계 자체는 저장 배열 검사만 하고 새 raycast/학습/solver를 실행하지 않습니다.

공통 복구의 `repair_worker_wall_sec`는 한 번의 prefix 비용이며 method별 참조 행을 더해 두 번 세지 않습니다.
조건마다 성공한 실행만의 품질 평균이므로 복구 성공률을 빼고 유리한 평균만 제시해서는 안 됩니다.
3 seeds와 단일 개발 지도, 고정 실행 순서, 공유 서버 시간으로 일반화·통계적 우위·통제된 speedup을 주장하지 않습니다.
새 제약용 MILP upper bound는 아직 없으므로 near-optimal gap도 보고하지 않습니다.

**2026-10-20 파일럿 27/27 반복 처리 완료:** 공통 feasible 초기값을 확보한 것은 18회입니다.
이 중 원래부터 feasible 9회, 복구 성공 9회입니다. LS/MAPPO 실제 실행은 각각 18개, 합계 36개이고,
나머지 9개 paired 반복의 method 슬롯 18개는 실행하지 않았습니다. 예기치 않은 worker 실패는 없습니다.

| 조건 | feasible 확보 / 3반복 | 처리 결과 |
|---|---:|---|
| baseline, cost_39, overlap_025 | 각각 3/3 | 원래 초기값 사용 |
| spacing_24, overlap_075, forbidden_1 | 각각 3/3 | 공통 복구 후 보정 |
| cost_38 | 0/3 | 고정 3대의 설치비용 하한 위반 |
| spacing_36 | 0/3 | 원래 anchor와 반경 4m의 두 센서 거리 상한 위반 |
| forbidden_6 | 0/3 | 지정 국소 탐색 한도 내 미발견. 불가능 증명 아님 |

성공한 6개 조건 모두 최종 목적값 J의 LS 평균이 MAPPO 평균보다 높았습니다.
6초 이전 관측만 사용하는 paired J 비교도 18쌍 모두 LS가 높았습니다. 이번 실행에서는 정시 J와
종료 후 best J가 같았지만, 실제시간은 별개입니다. 36개 method 실행 중 8개는 6초 soft limit를 넘었습니다.
조건별 평균 보정 시간은 LS 2.83~3.14초, MAPPO 5.85~6.17초였습니다.
이는 보정 구간의 공유 서버 단일 측정 반복이며 process/전이/공통 복구를 합친 speedup이 아닙니다.

예를 들어 junction 중복률 요구를 0.25에서 0.75로 올리면 LS의 최종 coverage는
99.877%에서 99.632%로 낮아지는 대신 해당 영역의 실제 이중 관측률은 0.522에서 0.815로 높아졌습니다.
이 예시는 목적과 필수 제약의 절충을 보여주며, 연속 추적이나 검출 성능 보장은 아닙니다.
MAPPO 18개는 각각 128 joint steps/1 PPO update뿐이므로 학습 수렴이나 장기 성능의 일반적 우열은 판단하지 않습니다.

실행 코드·source/input은 `frozen/`, 실행 후 확장한 정시 요약 집계 코드는 `report_source/`와
`validation_summary.json.report_source_sha256`로 구분합니다. `--report-only`는 원자료를 바꾸지 않습니다.
검증은 새 9개 테스트를 포함한 62개 중 61개 통과/전체 후보 parity opt-in 1개 skip,
문법 검사·기존 세 방법 dry-run 통과입니다. 축소 연결 점검은 `results/constrained_sensitivity_smoke_20261020/`입니다.

검증 명령:

```bash
PYTHONPATH=tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m unittest -v test_constrained_sensitivity test_feasibility_repair \
  test_constrained_coverage test_common_placement test_pipeline test_local_mappo \
  test_prototype_runtime test_placement_ui
```

## 이전 Prototype 진행 상태 (2026-10-14)

진행하던 일은 **새 지도에서 전이/보간 후 국소 보정한 배치를, 같은 설치 허용 공간의 전체 후보 MILP와 비교하는 것**입니다.
온라인 72개 실험은 이미 완료했고, 최근에는 MILP 후보 공간을 넓혀 기준선을 다시 계산했습니다.
현재 해당 학습/기준선 프로세스는 실행 중이지 않습니다. 이번 문서 갱신에서 학습/solve는 실행하지 않았습니다.

```text
오프라인 source library (4개 source 형상 × K1~6)
  → source별 pose/geometry 저장
  → 새 target 4개에 도로 상대 단독 전이 / 가능한 두-source 보간
  → K3,4의 초기값 18개: LS 18회 + MAPPO 3 seeds × 18 = 54회
  → target full cache 없이 on-demand 보정, 총 72개 온라인 결과 확정
  → 별도 오프라인 평가: target 전체 0.5m 격자 visibility/candidate 생성
  → GREEDY + 교환 탐색 + cut-MILP, 4개 target × K3,4 = 8개 기준선
  → 저장 pose의 실제 archive 포함 여부와 target별 관측 밀도 대조
  → 최신 비교표 18행, raw 비교표 72행 생성 완료
```

여기서 전이하는 것은 **배치 pose**입니다. MAPPO 사전학습 정책을 새 지도에 전이하는 모델이 아니며,
국소 MAPPO는 각 초기값/seed에서 새로 학습했습니다. source library의 기존 해들도 모두 최적 인증된 정답은 아닙니다.
source library를 전체 0.5m 격자로 다시 구축한 것은 아니고, 이번 full-grid 실행은 **target 기준선 4개 지도**에 해당합니다.

### 지금 볼 파일

| 자료 | 경로 / 의미 |
|---|---|
| 최신 전체 후보 기준선 8개 | [summary.csv](results/prototype_full_grid_20261012/summary.csv) |
| 초기값별 LS/MAPPO vs 새 기준선 18행 | [comparison.csv](results/prototype_full_grid_20261012/comparison_report/comparison.csv) |
| 온라인 seed별 원자료 비교 72행 | [raw_comparison.csv](results/prototype_full_grid_20261012/comparison_report/raw_comparison.csv) |
| 이번 집계 검증/평균 | [validation_summary.json](results/prototype_full_grid_20261012/comparison_report/validation_summary.json) |
| 완료 상태와 source/input hash | [manifest.json](results/prototype_full_grid_20261012/manifest.json) |
| 3D 기준선 예시 | [T자 420001/K3](results/prototype_full_grid_20261012/s420001/k3/view.html) |

IDE에 열려 있던 `results/prototype_target_full_20261012/comparison.csv`는 이전 incidence 기준선 비교입니다.
`prototype_cut_study_20261012/online_comparison.csv`도 이전 대표 격자 기준선입니다.
두 자료는 보존하되 현재 결과 해석은 위 full-grid 비교표를 먼저 봅니다.

### 현재 결론과 남은 작업

1. **배치 재사용과 보정은 구현·실행 완료:** 하지만 아직 개발용 4개 target의 실험이지 미사용 실도로 일반화 입증은 아닙니다.
2. **이번 온라인 배치의 후보 영역 문제는 해소:** 72개 모두 새 full-grid archive 안에 있고 target별 density도 일치합니다.
3. **MAPPO 우위는 없음:** 18개 초기값 모두 LS가 MAPPO 평균보다 높고 기록된 시간도 짧았습니다.
4. **기준선 품질 추가 보강 필요:** 300초에서 얻은 incumbent이지 최적값이 아닙니다. 전체 격자에서 이전보다 낮은 해가 나온 조건도 있습니다.
5. **다음 우선순위:** 이전 오프라인 해를 새 공간에서 검산해 더 좋은 feasible 초기해로 보존하는 독립 기준선 보강,
   이후 같은 계측 경계/반복 조건의 cold-start 시간 및 quality-time 비교입니다. 온라인 해를 몰래 solver 초기해로 넣지 않습니다.
6. **후속 연구:** target 정답을 보지 않는 prototype 선택 규칙, branch/side 센서 수가 달라 보간하지 못하는 사례,
   개발에 사용하지 않은 geometry/seed 평가가 남아 있습니다. 지금은 학습량만 늘리는 단계가 아닙니다.

## 전체 격자 packed 기준선 완료 (2026-10-12)

`prototype_full_grid.py`로 2026-10-12 20:41 KST에 새 실험을 시작했습니다.
범위는 420001/420002/420003/420004 × K=3,4, 총 8개입니다.
2026-10-12 **22:04:28 KST에 8/8 완료**했습니다. 전체 launcher 경과 시간은 5,003.48초(약 83분 23초)입니다.
이전 `prototype_cut_study_20261012`의 완료 8개와 다른 후보 공간의 새 실행입니다.

- geometry만으로 설치 가능한 0.5m XY cell 전체, Z=3~6m, 모든 허용 yaw/pitch를 생성합니다.
- 기존 MAPPO/LS 결과나 이전 candidate archive를 읽지 않습니다. 지도마다 새 run-local visibility cache를 생성합니다.
- `src/roadside/packed_candidates.py`는 pose별 mask를 바로 `packbits`하여 NPY memmap에 순차 기록합니다.
  전체 dense candidate-target 행렬을 RAM에 쌓지 않습니다. pruning과 mask dedup 모두 비활성화입니다.
- 4개 지도 후보 수는 각각 973,728 / 927,360 / 1,192,320 / 1,152,576입니다.
  mask 파일 크기는 dense bool의 약 1/8이며 Python/solver 메모리는 별도입니다.
- 각 K에서 packed GREEDY → 최대 30초 전체 후보 교환 탐색 → 300초 cut-MILP를 실행합니다.
- Gurobi 1 thread, 목표 gap=0, soft memory=90GB입니다. gap=0 설정이 최적성 인증을 보장하지 않습니다.
- 지도별 순차 worker이며 시작 전 가용 RAM 96GiB와 디스크 여유를 검사합니다.
  soft memory는 시스템 전체 강제 상한이 아니므로 OOM 방지를 절대 보장하지 않습니다.
- frozen source/config/scenario snapshot으로 실행합니다. worker 실패 시 다음 지도 실행을 멈추고 오류를 남깁니다.
- 결과는 공통 설치 제약과 독립 direct raycasting을 통과한 뒤 JSON/NPY/NPZ/3D HTML로 저장합니다.

실행 명령(현재 root는 존재하므로 재실행 시 새 이름 사용):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u prototype_full_grid.py \
  --output results/prototype_full_grid_20261012 \
  --k-values 3 4 --seconds 300 --swap-seconds 30 --execute
```

`--execute` 없으면 geometry 기반 계획만 출력하며 파일/학습/최적화를 생성하지 않습니다.
실제 실행은 터미널 종료와 분리한 background process로 시작했습니다.

```bash
tail -f results/prototype_full_grid_20261012/logs/s420001.log
cat results/prototype_full_grid_20261012/manifest.json
cat results/prototype_full_grid_20261012/s420001/manifest.json
```

전체 launcher 로그는 `/tmp/lidar-full-grid-20261012.log`입니다. 현재는 완료 로그이며 지도별 `s420002.log` 등도 보존되어 있습니다.
지도 manifest에는 visibility/candidate_masks/greedy/cut_milp/complete 단계와 후보 작성 수를 기록합니다.
`summary.csv/json`은 지도 완료 시 집계합니다. 해당 지도 K=3만 끝난 경우에는 `s<ID>/k3/result.json`을 먼저 확인합니다.

```text
results/prototype_full_grid_20261012/
  manifest.json / source/ / inputs/ / packages.txt / logs/
  summary.csv / summary.json
  s420001/ (s420002, s420003, s420004도 동일)
    manifest.json / config.yaml / scenario.json / cache/
    candidates/poses.npy / packed_masks.npy / targets.npy
    k3/ 및 k4/: result.json / poses.npy / coverage.npz / gurobi.log / view.html
```

시간은 cache 생성, candidate mask 작성, K별 GREEDY, swap/solver, 독립 검산,
지도 worker 전체 시간을 분리합니다. K=3과 K=4는 지도 전처리를 공유합니다.
ray 수는 최초 visibility 생성 계측이며 독립 검산 ray와 같은 항목으로 합산하지 않습니다.
fresh cache 실행이지만 기존 온라인 시간과 자동으로 동일 조건의 speedup 비교가 되는 것은 아닙니다.
온라인 18개 초기값 결과와 새 기준선의 비교표는 아래 재현 명령으로 생성했습니다.

### 전체 격자 최종 수치

| target | K | Coverage | 상대 gap | solver 상태 |
|---|---:|---:|---:|---|
| 420001 T자 | 3 | 99.837% | 0.163% | time limit |
| 420001 T자 | 4 | 99.939% | 0.061% | time limit |
| 420002 T자 | 3 | 99.908% | 0.092% | time limit |
| 420002 T자 | 4 | 99.918% | 0.082% | time limit |
| 420003 십자 | 3 | 93.619% | 6.064% | time limit |
| 420003 십자 | 4 | 99.886% | 0.114% | time limit |
| 420004 십자 | 3 | 97.873% | 2.174% | time limit |
| 420004 십자 | 4 | 99.279% | 0.726% | time limit |

8개 모두 `solver_status=9`, `discrete_optimal=false`입니다. 저장된 independent direct-raycast 검증은 8/8 통과입니다.
특히 `420003/K3`의 6.064% 상대 gap을 숨기거나 8개 전부를 exact optimum이라고 쓰면 안 됩니다.
위 gap은 solver 상한 U와 해당 실행 feasible 목적값 L의 `(U-L)/L`이며 coverage 오차 %p가 아닙니다.

**전체 후보가 늘었다고 제한시간 해가 항상 좋아지지는 않습니다.** 이전 대표 격자 cut-MILP보다
새 coverage가 낮은 조건은 5/8입니다. 예를 들어 420001/K4는 이전 100% → 이번 99.939%,
420004/K4는 99.856% → 99.279%입니다. feasible 공간의 최적값이 낮아진 것이 아니라,
후보 수/초기해/탐색 경로가 달라 같은 시간 안에 더 나쁜 incumbent로 끝난 것입니다.
이번 runner는 이전 해를 warm start로 가져오지 않았으므로 결과를 그대로 기록합니다.
다음 기준선 개선에서 검증된 이전 오프라인 해를 보존할 수 있으나, 아직 합친 수치로 원본 JSON을 수정하지 않았습니다.

### 기록된 계산량과 메모리

| target | cache 생성 | 후보 mask 작성 | 지도 worker 전체 (K3+K4 포함) | ray 수 (cache) | peak RSS |
|---|---:|---:|---:|---:|---:|
| 420001 | 9.37초 | 349.57초 | 1,090.25초 | 69,374,592 | 5.00 GiB |
| 420002 | 10.93초 | 425.30초 | 1,196.24초 | 87,575,040 | 5.93 GiB |
| 420003 | 12.01초 | 479.01초 | 1,273.71초 | 106,237,440 | 6.24 GiB |
| 420004 | 15.19초 | 599.10초 | 1,439.40초 | 138,977,280 | 6.81 GiB |

출처는 root manifest의 `worker_wall_sec`와 지도 manifest의 `candidate_stats`, `runtime`입니다.
peak RSS는 Linux 해당 worker 프로세스의 기록이지 서버 전체 사용량이나 미래 실행의 메모리 상한이 아닙니다.
실제 solver 시간은 약 300~314초입니다. callback/종료 처리 때문에 설정 300초를 조금 넘을 수 있습니다.
`optimization_sec`에는 swap/model 구성/solver 등이 포함되고 GREEDY/지도 전처리는 별도입니다.
이 비용과 LS/MAPPO 온라인 시간을 바로 나눈 수치는 통제된 speedup으로 보고하지 않습니다.

### 동일 archive 기준 온라인 비교

`prototype_full_grid_report.py`는 학습이나 solver 없이 완료 결과를 읽습니다.
실제 `candidates/poses.npy`에서 각 pose의 XYZ/yaw/pitch를 찾고 해당 packed mask를 합산합니다.
target 좌표 배열, target별 density, coverage, K 제약, 동일 초기값 hash와 seed 완전성을 확인합니다.
**기준선 8개와 온라인 72개 모두 실제 archive에 포함되며 저장 density/coverage와 일치**했습니다.
이는 이번에 수행한 저장 배열 검증이며 새 direct raycasting을 수행했다는 뜻은 아닙니다.

| 초기값 18개 평균 | Coverage | 새 기준선 incumbent 대비 차이 | 기록된 평균 온라인 시간 |
|---|---:|---:|---:|
| 초기 전이/보간 배치 | 95.661% | 별도 비교 | 별도 집계 |
| LS 보정 | 97.642% | -1.229%p | 3.91초 |
| MAPPO 보정 (3-seed 평균) | 97.087% | -1.784%p | 61.18초 |

초기값별 동일 가중치 기술통계라 지도/K 기준선은 초기값 수만큼 반복됩니다. 표본 SD는 MAPPO seed별 `ddof=1`입니다.
18개 중 LS가 새 기준선 incumbent보다 높은 조건은 4개, MAPPO 평균이 높은 조건은 3개입니다.
모두 허용 후보 내 배치이므로 이는 시간 제한 incumbent가 최적이 아니라는 증거이지 MILP 최적값을 능가한 결과가 아닙니다.
MAPPO는 최종 정책 한 번의 성능이 아니라 보정 중 보존한 best deployment를 비교한 것입니다.

초기값 선택에 민감합니다. 420003/K3에서 low 전이+LS는 92.588%, high 전이+LS는 86.397%이며
기준선은 93.619%입니다. 좋은 사례만 골라 전체 방법의 성능으로 보고하지 않습니다.
보간 가능한 경우는 최신 24개 target/K 중 2개뿐이며, 나머지 22개 실패를 0점이나 성공 표본으로 포함하지 않습니다.

재생성 명령(보고서 폴더의 파생 CSV/JSON만 갱신, 실험 원자료 변경 없음):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_full_grid_report.py \
  --root results/prototype_full_grid_20261012 \
  --study results/prototype_refinement_paired_20261011
.venv/bin/python tests/test_full_grid_report.py
```

생성: `comparison_report/comparison.csv`(18행), `raw_comparison.csv`(72행),
`validation_summary.json`(원자료 JSON hash/집계 코드 hash/평균/검증 범위).
이 도구는 현재 worktree의 이전 절대 경로 대신 지정한 root 아래 파일을 읽으므로 프로젝트 이동 후에도 집계할 수 있습니다.

검증: 새 packed/runner 테스트 5개(축소된 실제 raycast+K1 solve+검산 포함),
cut-MILP 테스트 4개(9개 solver/brute-force 대조 포함), 기존 pipeline 7개 및 문법 검사 통과.
새 4개 지도 dry-run도 통과했습니다. 이 테스트는 대규모 8개 문제의 완료나 최적성을 뜻하지 않습니다.

## 후보 공간 통일 구현 및 영역 검사 (2026-10-12)

최신 작업은 **공통 물리적 이산 후보 공간 정의와 기존 온라인 배치의 포함 여부 검사**입니다.
이 영역 검사 단계에서는 전체 visibility 생성/MILP를 실행하지 않았습니다. 이후 위 새 runner로 8개 재계산을 완료했습니다. 아래 기존 coverage/gap은
이전 `bin_representatives` 공간의 결과이며, 새 공간의 최적성 인증으로 사용하지 않습니다.

기존 기본 YAML과 과거 결과는 보존했습니다. 새 옵션은 다음과 같습니다.

```yaml
candidate:
  xy_domain: installable_grid
  xy_step_m: 2.0
  max_orientations_per_position: 0
```

- `xy_domain` 생략: 기존 `bin_representatives`, 2m bin마다 첫 설치 가능 cell을 유지합니다.
- `installable_grid`: `MapGeometry.is_installable()`을 만족하는 지도 내부 0.5m cell center를 모두 사용합니다.
- 높이 3~6m/1m, 상대 yaw -55~55도/5도, pitch -25~-5도/2.5도는 그대로입니다.
- `installation_base_positions()`와 `validate_installation_pose()`를 `src/environment.py`에 두었습니다.
  MILP/GREEDY 후보 builder는 전자를 사용하고, 국소 MAPPO/LS pose 검사는 후자를 사용합니다.
- `xy_step_m=2`는 새 full-grid 모드에서 후보 subsampling 간격이 아닙니다. 온라인 action 이동은 여전히 2m입니다.
- 10m anchor 반경, 이동 경로의 연결성은 온라인의 탐색 제한입니다. 모든 후보에 도달할 수 있다는 뜻은 아닙니다.
  온라인 배치가 전체 오프라인 허용 공간의 부분집합이 되도록 한 것이지 두 알고리즘의 탐색 절차를 같게 한 것은 아닙니다.
- mask dedup 전 공간을 기준으로 검사합니다. 동일 coverage mask dedup은 현재 비가중 union/at-most-K 목적값을
  보존하지만 모든 원래 pose를 archive에 남기지는 않습니다. 위치별 비용/설치 용량 제약을 추가하면 재검토해야 합니다.
- `xy_domain`은 config/cache hash에 포함됩니다. 이전 대표 격자 archive를 새 격자 결과로 재사용하면 안 됩니다.

영역은 **geometry/config만으로 먼저 생성**하고 나중에 기존 온라인 pose를 검사했습니다.
온라인 정답 pose, coverage 또는 MILP 결과를 이용해 영역을 확장하지 않았습니다.

| target | 이전 XYZ 후보 | 전체 격자 XYZ 후보 | yaw/pitch 포함 후보 수 (dedup 전) |
|---|---:|---:|---:|
| 420001 | 1,176 | 4,704 | 973,728 |
| 420002 | 564 | 4,480 | 927,360 |
| 420003 | 720 | 5,760 | 1,192,320 |
| 420004 | 688 | 5,568 | 1,152,576 |

[영역 검사 report.json](results/prototype_domain_audit_20261012/report.json):
기존 온라인 배치 **72/72가 full-grid XYZ/Z/yaw/pitch 조건을 통과**, 영역 밖 0건입니다.
동일 자료를 이전 XYZ 대표 격자로 검사하면 기존과 같은 8건이 영역 밖입니다.
이 검사는 최종 배치의 포함 여부이며 모든 학습 trajectory 재생, coverage 재검산, 최적성 검증은 아닙니다.
기존 full-grid 밖 경고를 과거 결과 파일에서 삭제하거나 기존 상한을 새 상한으로 바꾸지 않았습니다.

후보가 약 4~8.1배 늘어나며 모든 raw bool mask를 한 번에 보관하면 약 13.37~26.79 GiB입니다.
이는 mask 배열만의 이론 크기로, Python object/dedup/archive/solver 메모리를 포함한 피크 RAM 추정이 아닙니다.
이후 위 `prototype_full_grid.py`에 packed 저장과 cut-MILP를 연결했습니다.
현재 `prototype_offline.py --execute`는 기존 incidence solver 경로이므로 아래는 **dry-run만** 제시합니다.

실행한 검사 명령(학습/solve/cache 생성 없음, 새 output에 보고서와 config/시나리오 snapshot 저장):

```bash
.venv/bin/python prototype_domain_audit.py \
  --study results/prototype_refinement_paired_20261011 \
  --output results/prototype_domain_audit_20261012

.venv/bin/python prototype_offline.py \
  --config results/prototype_domain_audit_20261012/config.yaml \
  --scenario-json results/prototype_domain_audit_20261012/s420002.json \
  --xy-domain installable_grid --k-values 3 4 \
  --output results/prototype_full_grid_s420002
```

audit의 `--output`을 생략하면 파일을 쓰지 않고 출력만 합니다. 기존 output 디렉터리는 덮어쓰지 않습니다.
기본 설정 검증은 기존 MILP/GREEDY/MAPPO dry-run 세 가지 모두 통과했습니다.
`tests/test_installation_domain.py` 5 tests: 기존 후보 순서 유지, 전체 영역 포함/2m 이동 closure,
config hash 분리, 자세 grid 및 현재 yaw clamp 일치, pruning 없는 builder 경로를 검사했습니다.
기존 `tests/test_pipeline.py` 7개, `tests/test_prototype_runtime.py` 8개,
`tests/test_refinement_study.py` 5개도 통과하여 총 25개 테스트와 Python 문법 검사를 통과했습니다.
runtime/pipeline 테스트에는 작은 임시 raycast/PPO/solver 문제가 포함되며 대규모 실험 실행과 다릅니다.

새 영역 full archive/기준선 8개와 온라인 비교표는 완료했습니다.
아직 미완료인 것은 더 강한 feasible 초기해를 보존한 기준선 보강 및 동일 계측 범위의 반복 cold-map runtime 실험입니다.

작업 중 프로젝트 디렉터리가 `/home/robotserver/hj_ws_main`으로 이동했습니다.
이후 명령은 새 root에서 실행했습니다. 과거 JSON의 `/home/robotserver/hj_ws/...` 절대 경로는
자동 수정하지 않았으므로 과거 runner/viewer 재실행 전 경로를 확인해야 합니다.

## 이전 대표 격자 MILP 기준선 모델 개선 (2026-10-12)

기존 전체 후보 모델은 예를 들어 `420002/K=3`에서 행렬 비영 계수가 649,028,628개였습니다.
300초 제한 실행은 7개 문제에서 presolve 중 종료했고, 로그상 simplex iteration과 탐색 node 모두 0입니다.
저장된 GREEDY 초기해를 근사최적 정답으로 쓰지 않습니다. 420001/K=4만 100% coverage로 인증됐습니다.

새 opt-in 구현은 `src/methods/coverage_cut_milp.py`입니다. 기존 MILP 함수와 기본 설정은 변경하지 않았습니다.
**동일한 archive 후보와 target을 그대로 유지**하면서 큰 incidence 행렬을 처음부터 만들지 않는
submodular coverage cut formulation을 사용합니다. 새로운 연구 알고리즘의 독창성으로 주장하지 않습니다.
수식은 [Salvagnin의 submodular IP formulation, 식 (4)~(8)](https://www.dei.unipd.it/~salvagni/pdf/submod.pdf)에 해당하고,
정수해에서의 제약 생성은 [Gurobi lazy-constraint API](https://support.gurobi.com/hc/en-us/articles/360013197972-How-do-I-implement-lazy-constraints-in-Gurobi)를 사용합니다.

```text
A_i = candidate i의 관측 target 집합
f(S) = |union(i in S) A_i|
gain_i(S) = |A_i minus union(j in S) A_j|

maximize theta
sum_i x_i <= K
x_i binary, 0 <= theta <= target_count
theta <= f(S) + sum_i gain_i(S) * x_i    for every S
```

임의 배치 T의 coverage는 S에서 관측한 점과 T의 센서들이 추가 관측할 점을 합한 수보다 클 수 없습니다.
중복은 우변에서 여러 번 셀 수 있으므로 이 식은 upper bound입니다.
T=S에서는 gain 항이 0이어서 등호가 됩니다. 정수 후보해의 theta가 실제 f(T)보다 클 때
해당 T의 제약을 추가하면 원래의 이산 maximum-coverage 문제와 동등합니다.
모든 제약을 미리 만들지 않는다는 뜻이지 후보나 target을 줄이는 근사는 아닙니다.
정확한 formulation과 제한시간 내 최적성 인증은 별개입니다.

실행 순서:

1. 기존 `candidates.npz`의 packed masks를 읽습니다. 새 raycast cache나 후보는 만들지 않습니다.
2. 기존 MILP incumbent를 시작점으로 **전체 archive 후보의 1개 교환/추가 탐색**을 최대 30초 수행합니다.
3. 이 독립적 feasible 해를 lazy-cut MILP에 warm start로 넣고 1 thread/300초/gap0으로 풉니다.
4. 위반하는 정수 후보에만 coverage cut을 추가합니다. callback 오류는 종료·실패로 처리합니다.
5. 최고 feasible 배치의 coverage를 별도 direct raycasting으로 다시 검사합니다.

교환 탐색은 온라인 전이 결과를 읽지 않으며, 전체 archive에 접근하는 **오프라인 기준선용 heuristic**입니다.
반경 10m의 온라인 국소 탐색과 동일한 방법/계산 비용으로 설명하지 않습니다.
성능이 좋아졌다면 `swap.coverage_count`와 solver 후 objective를 구분하여 개선 출처를 확인해야 합니다.

출력의 `solver_best_bound`는 solver의 유한 bound(없으면 null),
`certified_upper_bound`는 이 값과 target 수 중 작은 값입니다. solver bound가 없으면 target 수만 사용하며
`bound_source=target_count_only`로 명시합니다. 저장 feasible 값 L과 이 상한 U의 `(U-L)/L`을
`certified_gap`에 기록합니다. `discrete_optimal`은 이 차이가 수치 허용치 내 0일 때만 참입니다.
이는 주어진 archive의 이산 문제에 대한 인증이며 연속 공간 최적성이나 실제 검출 성능 인증이 아닙니다.

대표 진단 명령:

```bash
.venv/bin/python prototype_milp_diagnostic.py \
  --output results/prototype_cut_milp_s420002_k3_20261012

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u prototype_milp_diagnostic.py \
  --baseline results/prototype_target_full_20261012/baselines/s420002 \
  --k 3 --seconds 300 --swap-seconds 30 \
  --output results/prototype_cut_milp_s420002_k3_20261012 --execute
```

4개 지도/K=3,4로 확대하는 runner는 `prototype_cut_study.py`입니다.
완료한 대표 진단은 archive/source hash와 예산을 확인한 뒤 재사용할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u prototype_cut_study.py \
  --output results/prototype_cut_study_20261012 \
  --reuse results/prototype_cut_milp_s420002_k3_20261012 --execute
```

`--execute`를 빼면 계획만 출력합니다. 기존 output은 덮어쓰지 않습니다.
대표 및 각 문제 폴더: `result.json`, `gurobi.log`, `poses.npy`, `coverage.npz`, 회전 가능한 `view.html`.
확대 runner root: `manifest.json`, `summary.csv/json`, worker별 `logs/`, 고정된 `source/`.
이번은 **archive 재사용 solver 진단**으로, 시간에 cold-map cache/후보 생성은 포함되지 않습니다.
기존 온라인 결과 8개의 후보 집합 차이도 해결한 것이 아닙니다. 대표 420002는 해당 차이가 없는 지도입니다.

검증: `tests/test_coverage_cut_milp.py`의 모든 부분집합 부등식 검사,
9개 작은 문제의 brute-force optimum 비교, padding/교환 탐색 검사를 통과했습니다.
전체 real-map 최적성 인증과 확대 결과는 manifest/result를 기준으로 확인합니다.

### 대표 진단과 전체 8개 완료 결과

`results/prototype_cut_milp_s420002_k3_20261012/result.json`의 완료 결과:

| 항목 | 값 |
|---|---:|
| 유지한 후보 수 | 116,700 |
| 이전 300초 incidence MILP coverage | 91.283% (17,844 targets) |
| 전체 후보 교환 탐색 후 | 99.918% (19,532 targets) |
| 교환 탐색 실제 시간 | 7.72초, 3 sweeps, local optimum 종료 |
| cut-MILP 종료 coverage | 99.918% (19,532 targets) |
| solver upper bound | 19,547.4508 targets |
| 상한 대비 상대 gap | 0.0791% |
| solver 시간 / 상태 | 300.01초 / time limit |
| 정확한 최적해 인증 | 아님 |
| 독립 direct raycasting 검산 | 통과 |

좋은 feasible 배치는 교환 탐색이 찾았고, MILP는 이를 개선하지 못했지만 유한 bound를 기록했습니다.
실행 로그에서 presolve 약 0.16초, 첫 root LP 약 0.07초로 실제 분기 탐색까지 진행했습니다.
대표 문제의 전체 최적화 시간은 약 310.25초이며 archive load 약 0.63초와 최종 검산/3D export는 별도입니다.
이 수치를 모든 지도나 cold-start 실행의 speedup으로 일반화하지 않습니다.

대표 결과 1개를 재사용하고 나머지 7개를 순차 계산하여 **2026-10-12 13:43:56 KST에 8/8 완료**했습니다.
아래 수치는 [summary.csv](results/prototype_cut_study_20261012/summary.csv)와 각 `result.json` 기준입니다.
gap은 `(U-L)/L × 100`이며 coverage 차이(%p)가 아닙니다.

| target / 형상 | K | 이전 coverage | 새 coverage | 상대 gap | 이산 최적성 확인 |
|---|---:|---:|---:|---:|---|
| 420001 / T자 | 3 | 89.748% | 99.810% | 0.190% | 아니오 |
| 420001 / T자 | 4 | 100.000% | 100.000% | 0% | 예 |
| 420002 / T자 | 3 | 91.283% | 99.918% | 0.079% | 아니오 |
| 420002 / T자 | 4 | 99.918% | 100.000% | 0% | 예 |
| 420003 / 십자 | 3 | 86.673% | 93.201% | 1.477% | 아니오 |
| 420003 / 십자 | 4 | 98.151% | 99.886% | 0.114% | 아니오 |
| 420004 / 십자 | 3 | 86.470% | 97.925% | 1.116% | 아니오 |
| 420004 / 십자 | 4 | 99.042% | 99.856% | 0.144% | 아니오 |

2개는 100% coverage이고 solver status=2입니다. 나머지 6개는 status=9(time limit)이지만
모두 유한 상한을 확보했습니다. 8개 모두 저장된 독립 direct-raycast 검증이 참입니다.
이 문서 갱신에서는 검증 기록과 수치를 읽었으며 최적화/학습/raycast를 재실행하지 않았습니다.

개선 출처도 구분합니다. 8개 중 6개는 교환 탐색 후 objective와 최종 objective가 같았습니다.
MILP가 추가로 개선한 것은 `420002/K4`(19,540 → 19,548 targets)와
`420004/K4`(24,906 → 24,924 targets)입니다. 따라서 결과 명칭은
**전체 후보 교환 탐색 + coverage-cut MILP 기준선**이며 개선 전체를 MILP 단독 성과로 설명하지 않습니다.

### 이전 대표 격자 비교표와 해석

| 자료 | 역할 |
|---|---|
| [summary.csv](results/prototype_cut_study_20261012/summary.csv) | 새 기준선 8개: coverage, 상한, gap, 개선 출처, 시간 |
| [online_comparison.csv](results/prototype_cut_study_20261012/online_comparison.csv) | 초기값 18개별 기존 온라인 결과와 새 기준선의 차이 |
| [paired_summary.csv](results/prototype_refinement_paired_20261011/paired_summary.csv) | 초기 배치, LS, MAPPO 3-seed 평균/표본 SD와 온라인 시간 |
| [manifest.json](results/prototype_cut_study_20261012/manifest.json) | 완료 상태, 실행 명령, 문제별 결과 경로, source hash |

**기존 `prototype_target_full_20261012/comparison.csv`는 덮어쓰지 않았습니다.**
이전 약한 기준선과의 차이를 보존한 역사 자료입니다. 이 절의 `online_comparison.csv` 역시 대표 격자 기록이며,
최신 전체 격자 비교에는 문서 앞부분의 `prototype_full_grid_20261012/comparison_report/comparison.csv`를 사용합니다.
각 기준선의 `result_json` 경로 옆 `view.html`에서 배치를 회전하며 볼 수 있습니다.
대표 진단을 재사용한 `420002/K3`의 원본은 별도 `prototype_cut_milp_s420002_k3_20261012/`에 있습니다.

18개 초기값을 동일 가중치로 집계하면 초기 95.661%, LS 97.642%, MAPPO 평균 97.087%입니다.
새 기준선 대비 평균 차이는 LS **-1.299%p**, MAPPO **-1.854%p**입니다.
동일 지도/K 기준선이 초기값 수만큼 반복되는 기술통계이며, 후보 영역 밖 결과도 포함합니다.
따라서 이 평균 차이는 최적성 gap이나 통계적 우위 검정이 아닙니다.
18개 조건 모두 LS가 MAPPO 3-seed 평균보다 높았고 기록된 평균 온라인 시간은 LS 3.91초,
MAPPO 61.18초입니다. 현재 구현·실행 예산에서 MAPPO의 추가 이점은 입증되지 않았습니다.

후보 영역 차이가 없는 `420002/K3/transfer_low` 사례는 초기 99.412%, LS 99.903%,
MAPPO 평균 99.615%, 새 기준선 99.918%입니다. LS와 기준선의 차이는 약 0.015%p입니다.
이 사례는 전이 후 적은 보정의 가능성을 보여주지만 모든 초기값·지도에서 같은 성능을 보장하지 않습니다.
보간은 최신 24개 target/K 중 2개에서만 가능했고, 단독 전이 대비 보간 우위도 입증하지 못했습니다.

후보 영역 검사는 이전 `online_domain.json`의 결과를 유지합니다. 새 solver도 같은 archive를 쓰므로
온라인 72개 중 8개(LS 1개, MAPPO 7개)의 XYZ 영역 불일치는 그대로입니다.
`local_search_inside_milp_domain`과 `all_mappo_trials_inside_milp_domain`이 거짓인 경우
coverage 차이를 동일 후보 공간의 최적해 대비 regret으로 해석하지 않습니다.
양수 차이가 생겨도 MILP 최적해를 능가했다는 뜻이 아닙니다. 새 기준선은 incumbent이며 상한과 다릅니다.

새 기준선 시간은 archive 재사용 조건입니다. cache/후보 생성, import, 검산/그림 저장의 포함 범위가
온라인 시간과 다르므로 이 표에는 speedup을 계산하지 않습니다. 통제된 cold-start 시간 비교는 **TBD**입니다.

아래는 대표 격자 진단 당시 정한 후속 순서입니다. 1번의 full-grid 구현/실행과 이번 72개 배치 검증은 완료했으며,
2~4번의 통제된 반복 성능 평가와 추가 ablation은 아직 미완료입니다.

1. 온라인 이동 가능 XYZ와 오프라인 후보 공간을 geometry 기반 규칙으로 통일합니다. 온라인 정답 pose를 후보에 몰래 추가하지 않습니다.
2. cache가 없는 새 지도에서 전처리·전이·보정·평가 시간을 같은 경계로 계측하고 ray 질의량도 비교합니다.
3. 초기값 선택 규칙을 target 정답과 독립적으로 고정하고, 개발에 쓰지 않은 형상에서 반복 seed 평가합니다.
4. 단독 전이/보간, LS/MAPPO, 보정 시간별 ablation으로 품질 회복과 계산 비용을 분리합니다.

이전 대표 격자 비교 CSV 재생성 명령(저장 결과만 집계하며 학습/solve하지 않음):

```bash
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

root = Path('results/prototype_cut_study_20261012')
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
def key(row):
    return int(row['scenario_id']), int(row['K'])
new_rows = read(root / 'summary.csv')
new = {key(row): row for row in new_rows}
paired = read(Path('results/prototype_refinement_paired_20261011/paired_summary.csv'))
old_rows = read(Path('results/prototype_target_full_20261012/comparison.csv'))
old = {(*key(row), row['initialization']): row for row in old_rows}
assert len(new_rows) == len(new) == 8
assert len(paired) == len(old_rows) == len(old) == 18
assert len({(*key(r), r['initialization']) for r in paired}) == 18
output = []
for row in paired:
    baseline = new[key(row)]
    previous = old[(*key(row), row['initialization'])]
    for field in ('initial_coverage_pct', 'local_search_coverage_pct', 'mappo_mean_pct'):
        assert abs(float(row[field]) - float(previous[field])) < 1e-9
    result = json.loads(Path(baseline['result_json']).read_text())
    assert result['validated_direct_raycast'] and not result['online_pose_warm_start']
    output.append({**row,
        'cut_reference_coverage_pct': baseline['coverage_pct'],
        'reference_relative_gap': baseline['relative_gap'],
        'reference_discrete_optimal': baseline['discrete_optimal'],
        'reference_upper_bound_targets': baseline['upper_bound'],
        'local_search_minus_reference_pp': float(row['local_search_coverage_pct']) - float(baseline['coverage_pct']),
        'mappo_minus_reference_pp': float(row['mappo_mean_pct']) - float(baseline['coverage_pct']),
        'local_search_inside_milp_domain': previous['local_search_inside_milp_domain'],
        'all_mappo_trials_inside_milp_domain': previous['all_mappo_trials_inside_milp_domain'],
        'reference_warm_optimization_sec': baseline['total_optimization_sec'],
        'reference_result_json': baseline['result_json']})
with (root / 'online_comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(output[0]))
    writer.writeheader()
    writer.writerows(output)
print('Validated and wrote', len(output), 'comparison rows; all 8 reference raycast validations passed.')
for field in ('initial_coverage_pct','local_search_coverage_pct','mappo_mean_pct','local_search_minus_reference_pp','mappo_minus_reference_pp'):
    print(field, sum(float(r[field]) for r in output)/len(output))
PY
```

## 동일 초기 배치의 국소 보정 비교 (2026-10-11)

`prototype_refinement_study.py`를 추가했습니다. 기존 library/MILP/geometry/PPO 구현은
바꾸지 않고, source 배치를 독립적으로 전이·보간한 뒤 동일 초기값으로 국소 탐색과 MAPPO를 비교합니다.
이는 **개발용 실험**이며 TR-C 게재 가능성, 새 지도 일반화, near-optimality를 입증한 결과는 아닙니다.
한국시간 2026-10-11 23:58:46~10-12 00:58:58에 72/72 작업을 완료했습니다. 실패 0건입니다.
현재 상태는 [manifest](results/prototype_refinement_paired_20261011/manifest.json),
완료 자료는 [index](results/prototype_refinement_paired_20261011/index.html)에서 확인합니다.

```bash
tail -f /tmp/lidar-refinement-paired-20261011.log
```

`tail` 종료는 실험을 중단하지 않습니다. 작업별 상세 로그는 결과 root의 `logs/`에 보존합니다.

### 먼저 확인한 대응 문제

최신 4개 target × K=1~6의 총 24개 사례에서 보간 가능은 2개,
불가능은 22개입니다. 22개 모두 두 source의 **도로 branch/설치선 side별 센서 배분 수가 다릅니다.**
전체 센서 수가 같아도 같은 설치선끼리 일대일 대응할 수 없다는 뜻입니다.
`diagnostics.json/csv`에 양쪽 설치선별 수와 `high_minus_low` 차이를 저장합니다.
대응 실패를 coverage=0으로 바꾸거나, 해당 실패 사례를 성공한 보간 표본에 포함하지 않습니다.
이 진단은 현재 엄격한 branch/side 대응 규칙에 대한 것이며 모든 보간 방법이 불가능하다는 증명은 아닙니다.

### 이번 비교 범위

| 항목 | 설정 |
|---|---|
| target | 420001, 420002, 420003, 420004: T자/십자 폭 및 교차점 변형 |
| K | 3, 4 |
| 초기값 | 단독 low 8개 + 단독 high 8개 + 가능한 보간 2개 = 18개 |
| 국소 탐색 | 초기값마다 결정적 best-improvement 1회, 총 18회 |
| MAPPO | 초기값마다 seed=0,1,2, 총 54회 |
| 총 작업 | 72회, fresh CPU process에서 순차 실행 |
| 공통 이동 범위 | 각 센서의 초기 XY 기준 반경 10m, 기존 11-action 제약 |
| 공통 보정 제한 | 60초 soft deadline |
| 추가 상한 | MAPPO 20,000 joint steps, 국소 탐색 100,000 neighbor evaluations |
| 초기값 선택 | coverage를 보고 하나를 고르지 않음. 18개를 각각 독립 평가 |

국소 탐색은 더 나은 1-agent 이웃이 없으면 일찍 종료합니다. MAPPO도 실제 step 수를 기록합니다.
60초는 동일한 **설정 상한**이지 같은 연산량·정확히 같은 실제시간을 보장하지 않습니다.
MAPPO의 잔여 PPO 업데이트, 마지막 평가, checkpoint 저장으로 제한시간을 넘길 수 있어
`soft_limit_overrun_sec`도 기록합니다. CPU 실행이며 PyTorch/BLAS/OpenMP는 1 thread로 설정합니다.
CPU affinity나 다른 라이브러리의 모든 내부 thread를 강제하지 않으며,
공유 서버의 다른 작업과 OS page cache까지 격리한 벤치마크는 아닙니다.

### Cold-map 조건과 시간 계측

1. source pose/config/geometry와 실행 코드를 새 결과 폴더로 복사해 고정합니다.
2. 새 worker는 source 배치에서 초기값을 다시 생성합니다. 단독 전이는 해당 source만 읽습니다.
3. target 전체 candidate visibility cache는 만들거나 로드하지 않습니다.
4. 보정 중 실제 방문 pose 및 관측 입력에 필요한 위치만 on-demand raycasting합니다.
   같은 worker 안의 중복 질의는 memoization하며 다른 방법/seed와 공유하지 않습니다.
5. 온라인 결과 확정 후에만 과거 전이 NPY와 초기값을 대조합니다. 초기값 선택에 과거 target coverage를 사용하지 않습니다.
6. 별도 direct raycasting으로 초기/최종 coverage, 설치 제약, anchor 반경, 초기값 보존을 검사합니다.

`online_runtime`은 source 읽기·기하 전이·보정(초기 scene 구성 포함)을 계측합니다.
프로세스 시작/import와 외부 hash 검사는 제외하고, 별도 `worker_wall_sec`에는 포함합니다.
최종 독립 검산은 `validation_runtime`, HTML export는 worker 전체 시간에만 들어갑니다.
ray 수/호출 수/배치 평가 수와 quality-time trace도 보관합니다. 이 수들은 FLOPs가 아닙니다.
기존 full-MILP 시간과 바로 나눠 speedup을 주장하지 않습니다. 동일 target 기준선은 완료됐지만 통제된 시간 비교는 후속입니다.

### 실행 명령과 자료

계획 확인은 결과를 생성하지 않습니다.

```bash
.venv/bin/python prototype_refinement_study.py \
  --output results/prototype_refinement_paired_20261011

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_refinement_study.py \
  --library results/prototype_library_shapes_20261008 \
  --output results/prototype_refinement_paired_20261011 \
  --k-values 3 4 --seeds 0 1 2 --seconds 60 --radius-m 10 \
  --steps 20000 --max-evaluations 100000 --execute
```

기존 output root가 있으면 실패합니다. 재실행은 새 root를 사용하세요. 완전한 이어학습 기능은 없습니다.
입력 library는 ignored 결과 경로이므로 clone만으로 포함되지 않습니다. 앞선 library 생성이 필요합니다.

```text
results/prototype_refinement_paired_20261011/
  manifest.json                 실행 상태, job별 완료/실패, 설정, 입력/source hash
  diagnostics.json / csv        24개 target/K의 대응 분석
  source/ / inputs/             실행 코드 및 source pose/config/geometry snapshot
  validation_references/        온라인 결정 후 검증 전용 이전 초기 배치
  packages.txt                  실행 의존성
  logs/                         작업별 로그
  jobs/<job>/
    initial.npy / deployment.npy / coverage.npz / result.json / view.html
    mappo/                      MAPPO 작업만: 정책, NPZ, best/final-policy 배치
  raw_results.csv               작업별 실제 시간·coverage·ray 수·step 수
  paired_summary.csv            동일 초기값 LS vs MAPPO mean/표본 SD
  summary.json / index.html     현재 집계 및 회전 가능한 3D 목록
```

집계는 완료 작업만 수치로 표시합니다. 실패/대기는 manifest에서 확인하며 0점 처리하지 않습니다.
paired 표는 LS 1회와 요청한 모든 MAPPO seed가 완료되고 초기 pose hash가 같을 때만 생성합니다.
MAPPO 대표값은 탐색/평가 중 최고 배치이며 최종 deterministic policy 성능은 별도 column입니다.
완료한 18개 초기값의 단순 평균은 다음과 같습니다. MAPPO는 각 초기값의 3-seed 평균을 집계합니다.

| 방법 | 평균 coverage | 평균 online 시간 |
|---|---:|---:|
| 전이/보간 초기 배치 | 95.661% | 별도 집계 필요 |
| 국소 탐색 보정 | 97.642% | 3.91초 |
| MAPPO 보정 | 97.087% | 61.18초 |

18개 조건 모두 국소 탐색이 MAPPO 평균보다 높았습니다. MAPPO 우위의 증거가 아니며,
시간은 공유 서버에서 측정한 실행값입니다. 이전 대표 격자와의 차이는 `online_comparison.csv`,
최신 전체 격자와의 차이는 `prototype_full_grid_20261012/comparison_report/comparison.csv`에 정리했습니다.
이후 전체 격자 archive의 pose 포함과 density 일치를 검증했습니다. 공정한 반복 cold-start 시간 비교는 아직 **TBD**입니다.
위 수치는 `paired_summary.csv`와 `summary.json`을 기준으로 재계산했습니다.

사전 점검 `results/prototype_refinement_smoke8_20261011/`은 6/6 작업 완료입니다.
세 MAPPO 작업 각각 8 joint steps/PPO update 1회와 최종 direct 검산을 확인했습니다.
앞선 2초 제한 점검은 초기화 중 시간이 소진되어 MAPPO 0 steps였으므로 학습 검증으로 사용하지 않습니다.
8-step 점검도 성능 비교나 수렴 근거는 아닙니다. 재현 명령(다시 실행할 때는 새 root):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python prototype_refinement_study.py \
  --output results/prototype_refinement_smoke8_20261011 \
  --scenario-ids 420003 --k-values 4 --seeds 0 \
  --seconds 30 --steps 8 --max-evaluations 10 --execute
```

검증 명령:

```bash
.venv/bin/python tests/test_refinement_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python tests/test_prototype_runtime.py
```

## 이전 incidence MILP 기준선과 후보 archive (2026-10-12)

`prototype_target_baseline.py`가 완료된 보정 실험을 고정한 뒤 동일한 4개 target/K=3,4의
8개 기준선을 생성했습니다. 한국시간 10월 12일 03:29:20에 완료했으며
`results/prototype_target_full_20261012/manifest.json`에 기록되어 있습니다.
완료는 계산·저장 완료입니다. 7개는 전처리 시간 초과로 GREEDY 초기해만 반환했고 유한 bound가 없습니다.

- 온라인 실험의 source snapshot으로 `prototype_offline.py`를 실행합니다. 기존 코드/라이브러리는 변경하지 않습니다.
- 기존 2m XY 구간별 대표 후보, Z=3~6m, 전체 yaw/pitch를 사용합니다. top-M=0이며 동일 coverage mask dedup은 유지합니다.
- target마다 새 cache와 전체 candidate archive를 생성합니다. 두 K는 같은 지도 전처리를 공유합니다.
- 온라인 배치/coverage를 solver warm start에 넣지 않습니다. 기존 GREEDY warm start만 사용합니다.
- Gurobi 1 thread, K별 300초, 목표 gap=0, soft memory=90GB, sparse target batch=128입니다.
- 지도별 순차 실행이며 시작 전 가용 RAM이 96GiB 미만이면 실패시켜 추가 실행을 막습니다.
  이는 실행 중 시스템 전체 메모리를 강제 제한하는 기능은 아닙니다.
- 시간 제한 결과는 `best_known`, 최적성 인증을 통과한 결과만 `discrete_optimal`로 기록합니다.
- 온라인 결과의 hash를 고정하고, 집계 시 target 좌표 배열까지 동일한지 검사합니다.

**중요한 후보 공간 차이:** 보정은 설치 가능한 0.5m cell 위에서 2m 단위로 이동하지만,
MILP는 2m bin마다 첫 설치 가능 cell을 대표로 남깁니다. 현재 온라인 72개 결과 중
8개(MAPPO 7개, 국소 탐색 1개)의 XYZ가 이 MILP 대표 후보 집합 밖에 있습니다.
물리 제약 위반은 아니며 `online_domain.json`에 pose별 검사를 남깁니다.
비교 CSV의 `local_search_inside_milp_domain`, `all_mappo_trials_inside_milp_domain`도 확인하세요.
따라서 이 기준선을 모든 온라인 행동에 대한 전역 upper bound라고 부르지 않습니다.
이것은 이전 incidence archive의 한계입니다. 이후 full-grid 배치 포함과 유한 bound를 확보했으며,
최신 인증 수준은 앞부분의 full-grid gap 표를 사용합니다. 해당 실행도 exact optimum은 아닙니다.

재현 명령(실제 실행에는 --execute 필요, 기존 root 재사용 금지):

```bash
.venv/bin/python prototype_target_baseline.py \
  --output results/prototype_target_full_20261012

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u prototype_target_baseline.py \
  --study results/prototype_refinement_paired_20261011 \
  --output results/prototype_target_full_20261012 --seconds 300 --execute

tail -f /tmp/lidar-target-full-20261012.log
```

각 지도는 실제 offline CLI dry-run을 먼저 통과한 뒤 계산합니다.
결과 root의 `baselines/s<ID>/`에는 config, mesh, visibility cache, candidates.npz,
각 K의 pose/result/validation/Gurobi log가 저장됩니다.
`comparison.csv/json`은 초기값별 MILP와 LS/MAPPO coverage 차이(%p),
`baselines.csv`는 bound/gap/status/후보 수/시간을 저장하며 `index.html`에서 확인할 수 있습니다.
실행 중에는 완료한 지도만 집계합니다. 미완료를 0점으로 채우지 않습니다.

시간은 다음과 같이 구분하며 하나의 speedup 수치로 바로 해석하지 않습니다.

| 필드 | 포함 범위 |
|---|---|
| candidate_build_sec | 새 visibility cache와 전체 후보 생성 |
| k_work_sec | 해당 K GREEDY warm start, MILP, 최종 공통 evaluator 검산 등 |
| candidate_plus_k_work_sec | 위 두 시간의 합. 해당 K를 cold-start로 계산하는 비용의 일부 집계 |
| solver_sec | solver가 기록한 시간 |
| map_worker_wall_sec | import, archive 저장, 두 K 계산 등을 모두 포함한 지도별 전체 worker 시간 |

`candidate_plus_k_work_sec`에는 candidate archive 직렬화/import가 빠져 있고 최종 검산은 포함됩니다.
온라인 `online_sec`와 정확히 같은 계측 구간이 아니므로 통제된 시간 절감 배율로 주장하면 안 됩니다.
이 이전 baseline runner는 ray 개수를 계측하지 않습니다. 새 full-grid runner는 cache 생성 ray 수를 기록하지만
동일 계측 범위의 반복 비교는 아직 TBD입니다.

검증: `python tests/test_target_baseline.py` 5 tests, 기존 pipeline 7 tests 및 문법 검사 통과.
이 절은 이전 incidence 모델 실행 기록입니다. 이후 같은 archive를 사용한 coverage-cut 실행은
8개 모두 유한 상한을 확보했습니다. 이후 전체 격자 기준선과 온라인 배치 포함 검증도 완료했습니다.
최신 표는 문서 앞부분을 사용하며 통제된 반복 시간 비교와 최적성 보강은 아직 미완료입니다.

## 라이브러리 형상 확장 (2026-10-08)

기존 직선/T자/십자 7m·14m 라이브러리는 보존했습니다. 이번 실행은
`configs/library_expansion.json`의 T자·십자 형상 각각 두 폭을 추가하고,
별도로 정의한 새 지도에 단독 전이 두 개와 설치 구간 비율 보간을 적용합니다.
**2026-10-08 18:03:44~20:38:37 KST에 완료**했습니다. 전체 경과 시간은 9,293.060초
(약 2시간 34분 53초)입니다. `manifest.status=complete`, `failed_transfers=0`입니다.
여기서 complete는 계획된 처리를 끝냈다는 뜻이며 모든 보간이 가능하거나 모든 MILP가 최적이라는 뜻은 아닙니다.
launcher 로그는 `/tmp/lidar-library-shapes-20261008.log`이며 임시 파일입니다.
보존용 지도별 상세 로그는 결과 root의 `logs/`와 각 K의 `gurobi.log`입니다.

먼저 볼 자료:

- [최신 결과 및 회전 가능한 3D 목록](results/prototype_library_shapes_20261008/index.html)
- [완료 manifest 및 전체 요약 수치](results/prototype_library_shapes_20261008/manifest.json)
- [실제로 사용한 source/target 지도 명세](results/prototype_library_shapes_20261008/suite.json)
- [보간 성공 예: 십자 폭 12m, K=4](results/prototype_library_shapes_20261008/transfers/s420004/k4/interpolation.html)
- [위 예의 센서 대응·보간 좌표·실제 수치](results/prototype_library_shapes_20261008/transfers/s420004/k4/report.json)

### 지도와 실행 범위

| 구분 | ID | topology | 폭(m) | 교차점 x,y(m) |
|---|---:|---|---:|---|
| library | 410001 | T_JUNCTION | 7 | 35,55 |
| library | 410002 | T_JUNCTION | 14 | 35,55 |
| library | 410003 | CROSS | 7 | 40,60 |
| library | 410004 | CROSS | 14 | 40,60 |
| 새 지도 | 420001 | T_JUNCTION | 9 | 37,57 |
| 새 지도 | 420002 | T_JUNCTION | 12 | 33,53 |
| 새 지도 | 420003 | CROSS | 9 | 42,58 |
| 새 지도 | 420004 | CROSS | 12 | 38,62 |

모두 100m x 100m synthetic development 맵이며 실제 OSM 형상을 가져온 것이 아닙니다.
source/target ID 및 geometry 중복을 거부합니다. 기존 4개 형상 파일럿과 다른 지도입니다.
T자는 410001/410002, 십자는 410003/410004를 원본 쌍으로 명시적으로 사용했습니다.
자동 유사 지도 검색이나 도시 도로망 분해를 수행한 것은 아닙니다. 차로 수는 모두 2로 고정했습니다.

- 오프라인: 기존 2m XY bin, 높이 3~6m, yaw 5도/pitch 2.5도 격자에서 top-M을 해제합니다.
- 후보별 packed coverage mask와 pose를 `candidates.npz`, geometry를 `geometry.json`에 저장합니다.
- 순수 union 목적의 동일 mask dedup은 유지합니다. 연속 공간의 모든 pose를 열거하는 것은 아닙니다.
- 지도별 후보를 한 번 생성하고 K=1~6 배치를 저장합니다. at-most-K이며 실제 설치 수를 별도 기록합니다.
- Gurobi 1 thread, 문제당 300초, 목표 gap 0, soft memory 90GB, target batch 128입니다.
  solver 제한은 전체 후보 생성/전처리를 포함한 wall-clock 제한이 아닙니다. 지도는 순차 실행합니다.
- 각 가족의 library 두 개가 끝나면 해당 새 지도에서 K별 전이를 수행합니다.
- 보간은 `prototype_segments.py`의 `(branch,side)` 대응, 설치 구간 비율 u, 상대 자세를 사용합니다.
  폭으로 alpha를 구하고, 새 교차점/가지 길이는 target 설치 구간에 반영합니다.
- source들의 실제 센서 수 또는 설치선 대응이 다르면 `unavailable`로 기록합니다.
  별도 설치선 제약을 자동 추가하거나 다른 해로 몰래 바꾸지 않습니다.
- 투영 실패는 `failed`로 기록하고 다른 조합을 계속 처리합니다. 실패가 있으면 최종 상태는
  `complete_with_failures`입니다. unavailable은 0% coverage가 아닙니다.
- 새 지도에는 전체 visibility cache/후보 mask를 생성하지 않고 전이한 pose만 direct raycast합니다.
  세 초기화를 각각 평가하며 최고 초기화만 남기는 실험이 아닙니다.
- 이번 실행에는 MAPPO, 새 지도 전체 후보 MILP, 성능 우위 검증이 포함되지 않습니다.
  배치 전이 결과와 library 인증 상태를 먼저 확인한 뒤 진행합니다.

### 실제 후보 수와 오프라인 비용

후보 수는 완전히 같은 coverage mask를 제거한 뒤의 값입니다. coverage edge는 후보가 관측하는
target의 incidence 수이며 ray 발사 횟수나 FLOPs가 아닙니다.

| source ID | target 수 | XYZ base 수 | orientation 후보 원래 수 | dedup 후 후보 수 | coverage edges | 후보 생성 초 | source 전체 초 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 410001 | 10,992 | 588 | 121,716 | 121,600 | 354,726,846 | 36.564 | 1,260.909 |
| 410002 | 23,256 | 556 | 115,092 | 115,044 | 752,911,412 | 60.337 | 1,702.820 |
| 410003 | 13,980 | 1,472 | 304,704 | 304,405 | 972,595,847 | 92.275 | 2,699.757 |
| 410004 | 29,184 | 688 | 142,416 | 142,358 | 980,673,380 | 84.347 | 3,604.143 |

출처는 각 `library/s<ID>/manifest.json`의 `candidate_stats`, `candidate_build_sec`,
`total_wall_time_sec`입니다. 전체 시간에는 후보 압축 저장, K별 warm start/행렬 구성/solver/검증/저장이
들어가므로 `6 × 300초`와 같지 않습니다. 300초는 hard process timeout이 아니며 실제 기록된
시간 제한 solve는 약 305~387초였습니다. 전체 실행에는 전이/3D export/프로세스 시작 비용도 더해집니다.

### K별 library 배치 결과

단위는 원본 지도에서의 coverage %입니다. 새 지도에서의 성능과 구분합니다.

| source ID | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |
|---:|---:|---:|---:|---:|---:|---:|
| 410001 T자 7m | 58.843 | 86.745 | 99.390 | 99.918 | 100.000 | 100.000 |
| 410002 T자 14m | 57.847 | 77.838 | 90.540 | 99.914 | 100.000 | 100.000 |
| 410003 십자 7m | 55.086 | 75.114 | 94.371 | 98.577 | 99.871 | 100.000 |
| 410004 십자 14m | 51.525 | 73.784 | 86.914 | 98.962 | 99.627 | 99.931 |

| 인증 / 종료 경로 | 개수 | 의미 |
|---|---:|---|
| `ANALYTIC_EXACT_K1` / `discrete_optimal` | 4 | 선언된 전체 이산 후보 중 단일 최대 coverage, solver 생략 |
| `ANALYTIC_FULL_COVERAGE` / `discrete_optimal` | 5 | feasible 배치가 100%에 도달해 coverage 상한 인증, solver 생략 |
| status 9 / `best_known` | 15 | 시간 제한. `best_bound=null`, `mip_gap=null`, 최적성 미인증 |

따라서 24개 모두를 "MILP로 증명한 최적 정답"이라고 부르면 안 됩니다.
9개는 분석적으로 이산 목적값을 인증했고, 실제 Gurobi를 호출한 15개는 시간 제한에서 끝났습니다.
`null`은 gap 0이라는 뜻이 아니며 이 metadata만으로 solver 내부 어느 단계에서 지연됐는지 단정하지 않습니다.

T자 두 source의 K=6 결과는 실제로 **5대**입니다. 나머지는 각 budget과 같은 대수를 저장했습니다.
at-most-K이므로 정상입니다. T자에서 5대로 100%를 달성했더라도 K=4의 최적성이 미인증이므로
"100%에 필요한 최소 센서 수가 5대"라는 증명은 아닙니다.

### 새 지도 전이와 보간 결과 전체

각 새 지도에서 K=1~6, `transfer_low`/`transfer_high`/`interpolation` 세 슬롯을 처리했습니다.
총 72개 슬롯 중 **50개 배치 저장(단독 전이 48 + 보간 2), 보간 불가 22, 실행 실패 0**입니다.
아래 `불가`는 coverage 0이 아니라 대응하는 배치 자체를 생성하지 않았다는 뜻입니다.
low/high는 각각 같은 topology의 7m/14m source이며, 셋 중 최고만 골라 저장하지 않았습니다.

| 새 지도 ID | K | Low 전이 % | High 전이 % | 보간 % | 전이·평가 기록 시간(초) |
|---:|---:|---:|---:|---:|---:|
| 420001 | 1 | 53.255 | 54.875 | 불가 | 2.719 |
| 420001 | 2 | 80.831 | 76.973 | 불가 | 0.400 |
| 420001 | 3 | 93.016 | 88.629 | 불가 | 0.567 |
| 420001 | 4 | 99.824 | 99.824 | 불가 | 0.445 |
| 420001 | 5 | 99.939 | 99.959 | 불가 | 0.459 |
| 420001 | 6 | 99.939 | 99.959 | 불가 | 0.461 |
| 420002 | 1 | 57.019 | 59.157 | 불가 | 0.386 |
| 420002 | 2 | 89.216 | 80.387 | 불가 | 0.402 |
| 420002 | 3 | 99.412 | 90.782 | 불가 | 0.412 |
| 420002 | 4 | 99.678 | 99.903 | 불가 | 0.546 |
| 420002 | 5 | 99.785 | 100.000 | 불가 | 0.434 |
| 420002 | 6 | 99.785 | 100.000 | 불가 | 0.437 |
| 420003 | 1 | 52.619 | 54.104 | 불가 | 0.547 |
| 420003 | 2 | 73.032 | 73.157 | 불가 | 0.465 |
| 420003 | 3 | 92.382 | 84.239 | 불가 | 0.475 |
| 420003 | 4 | 98.390 | 95.663 | 97.755 | 1.858 |
| 420003 | 5 | 99.870 | 98.411 | 불가 | 0.489 |
| 420003 | 6 | 99.967 | 98.791 | 불가 | 0.504 |
| 420004 | 1 | 52.179 | 52.312 | 불가 | 0.460 |
| 420004 | 2 | 75.076 | 74.475 | 불가 | 0.463 |
| 420004 | 3 | 97.548 | 86.034 | 불가 | 0.599 |
| 420004 | 4 | 99.331 | 99.720 | 99.764 | 0.610 |
| 420004 | 5 | 99.491 | 99.772 | 불가 | 0.506 |
| 420004 | 6 | 99.595 | 99.916 | 불가 | 0.512 |

target 수는 각각 420001=14,748, 420002=19,548, 420003=18,444, 420004=24,960입니다.
각 행은 같은 target 집합으로 비교하지만 서로 다른 지도의 백분율을 같은 관측 개수로 해석하지 않습니다.
출처는 `transfers/s<ID>/k<K>/report.json`입니다. 시간은 해당 report의 `runtime.elapsed_sec`입니다.

### 실제 보간 규칙과 불가 원인

실행 함수는 `prototype_segments.prepare_segment_placements()`입니다.

1. 공통 설치 판정의 격자에서 도로 가지와 설치선 side별 종방향 구간을 추출합니다.
2. 원본 센서를 해당 구간의 비율 좌표 `u`와 설치선 횡편차 `e`, 높이, 상대 yaw/pitch로 표현합니다.
3. low 배치를 high 설치 구간으로 옮긴 뒤 같은 `(branch,side)` 안에서 Hungarian 대응을 구합니다.
4. 대응 센서의 `u/e/z/pitch`는 선형, 상대 yaw는 최단 원호로 보간합니다.
5. 새 지도 구간에서 좌표를 복원하고 기존 허용 XYZ/yaw/pitch 격자로 투영합니다.
6. 설치 가능 여부·자세·budget·완전 중복 pose를 검사하고 그 배치만 direct raycast합니다.

```text
u = (s - s_min) / (s_max - s_min)
alpha = (w_target - 7) / (14 - 7)
u_target = (1-alpha) * u_low + alpha * u_high
s_target = s_min_target + u_target * (s_max_target - s_min_target)
e_target = (1-alpha) * e_low + alpha * e_high
d_target = side_sign * (w_target/2 + sidewalk_width/2 + e_target)
yaw_relative_target = yaw_low + alpha * wrap_to_180(yaw_high - yaw_low)
```

폭 9m는 alpha=2/7, 12m는 5/7입니다. 교차점 위치를 alpha로 보간하는 것이 아니라
**새 지도의 실제 설치 구간 길이와 원점에 배치를 복원**합니다. source 두 개의 교차점은 같은 위치입니다.
비대칭 source에서는 원점 기준 반사가 도로 대칭을 보장하지 않으므로 반사 대응을 허용하지 않습니다.
이 규칙은 구간 안의 종방향 위치를 보존할 뿐 coverage·전역 최적성·안정적인 대응을 보장하지 않습니다.

이번 보간 불가 22개는 모두 `No same-installation-side correspondence`로 기록됐습니다.
단순히 폭이 달라져 투영에 실패한 것이 아니라, 독립 배치의 **가지·side별 센서 수가 달라**
현재 허용하는 일대일 대응을 구성할 수 없었습니다. 예를 들어 K=3은 다음과 같습니다.

| source 쌍 | 7m의 (branch,side) | 14m의 (branch,side) |
|---|---|---|
| T자 | south/right, south/left, east/right | west/left, south/left, east/right |
| 십자 | south/right, north/left, east/right | west/left, north/left, east/right |

서로 다른 가지의 센서를 억지로 연결하지 않았습니다. 십자 K=4에서는 양쪽 모두
west/left, south/right, north/left, east/right에 한 대씩 있어 두 target의 보간이 가능했습니다.
T자 K=4의 14m 배치는 south/left에 두 대이므로 과거 "그룹당 한 대" 보조 생성기를
모든 K에 그대로 적용할 수 없습니다.

성공한 보간도 420003/K4에서는 최고 단독 전이보다 약 **0.635%p 낮고**,
420004/K4에서는 약 **0.044%p 높습니다**. 두 성공 사례만으로 보간 우위를 주장할 수 없으며
22개 불가를 제외한 평균만 제시해서는 안 됩니다. 독립 배치 하나씩만 보관한 library의
대응 가능성 한계가 드러난 결과로 해석합니다.

### Runtime 및 검증 해석

24개 target/K report의 전이·평가 기록 시간 합은 **15.156초**, 범위는 **0.386~2.719초**입니다.
이는 한 행의 가능한 초기 배치 2~3개를 함께 처리한 시간이며 센서 한 대나 보간만의 시간이 아닙니다.
geometry/구간 추출·대응·투영, 새 raycasting scene, 직접 coverage 평가 및 일부 결과 배열 저장을 포함합니다.
prototype 파일 로드·프로세스 시작·3D HTML export는 포함하지 않습니다. 같은 프로세스에서 순차 측정했으며
OS cache나 import 초기화 상태를 매번 지운 반복 runtime benchmark도 아닙니다.

50개 배치 평가에서 실제 제출 ray는 총 **1,526,880개**로 기록됐습니다.
target 전역 coverage cache나 모든 orientation mask를 생성·로드하지 않았습니다.
그렇다고 "지도 관련 계산이 전혀 없다"는 뜻은 아닙니다. 설치 구간·mesh·도로 target은 새로 생성합니다.
source offline 2시간 35분과 이 15초는 서로 다른 지도·작업 범위이므로 두 값을 나눠 speedup이라고 부르면 안 됩니다.
이 라이브러리 생성 실행 자체에는 target full-candidate MILP/국소 MAPPO가 없습니다.
이후 K3,4에서 3-seed 보정 및 full-grid 기준선 비교를 완료했습니다(문서 앞부분).
동등 하드웨어 조건의 반복 end-to-end 비교는 **TBD**입니다.

실행 중 library는 선택 mask union과 공통 evaluator, canonical 좌표 round-trip, solver 목적값을 검사했고,
전이 결과는 설치 제약 검사와 직접 mesh raycasting을 거쳐 저장했습니다.
2026-10-09 문서 갱신에서는 다음을 읽기 전용으로 추가 확인했습니다.

- source candidate archive 4개, K별 library result 24개, target/K report 24개.
- 성공한 50개 pose NPY의 shape/유한값 및 50개 coverage NPZ의 density와 JSON 수치 일치.
- 성공한 50개 3D HTML 및 참조 pose 파일 존재, 누락 0건.
- 이번 문서 작업에서 학습·solver·전역 raycast cache를 다시 실행하지 않았습니다.

이는 파일·계산 경로의 일관성 확인이지 모든 후보 mask의 독립 재계산, 정책 수렴, 최적성 증명이 아닙니다.

### 재현 명령과 파일

계획 확인(`--execute` 없으면 파일 생성/solve 없음):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_library.py \
  --output results/prototype_library_shapes_20261008 --k-values 1 2 3 4 5 6
```

실제 실행(기존 output root가 있으면 거부하므로 재실행에는 새 이름 사용):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u prototype_library.py \
  --output results/prototype_library_shapes_20261008 --k-values 1 2 3 4 5 6 \
  --solver-seconds 300 --execute
```

출력 구조:

```text
results/prototype_library_shapes_20261008/
  manifest.json                 # 현재 작업, 완료 library, 전이 상태, source hash
  suite.json / inputs/ / source/
  logs/s410001.log ...           # 지도별 후보 생성 및 MILP 로그
  library/s410001/ ...           # 전체 후보 archive/cache/geometry 및 k1~k6
  transfers/s420001/k1/ ...      # 독립 전이/보간 pose, coverage, report, 3D HTML
  index.html                    # 생성된 전이 결과와 불가/실패 사유
```

300초 내 해는 인증값에 따라 `best_known` 또는 `discrete_optimal`로 기록합니다.
목표 gap=0 설정 자체는 최적성 증명이 아닙니다. 기존 library는 참조 자료로 보존하며
이번 확장 runner가 자동 검색하거나 합쳐 재최적화하지 않습니다.
프로세스 중단 후 optimizer checkpoint를 이어받는 기능은 없습니다.
새 실행에는 예를 들어 `--output results/prototype_library_shapes_repeat`를 사용합니다.
정확한 당시 config와 Python source는 결과 root의 `source/`와 `source_hashes`로 구분합니다.
현재 코드를 고친 뒤 같은 명령을 실행하는 것만으로 당시 조건이 보존되는 것은 아닙니다.

파일의 역할:

- `library/s<ID>/candidates.npz`: `poses[M,5]`, `packed_masks[M,ceil(N/8)]`, `target_points[N,3]`.
- `library/s<ID>/cache/`: 원본 지도 LOS cache. 새 target의 online 전이에 공급하지 않습니다.
- `library/s<ID>/geometry.json`, `config.yaml`, `mesh.npz`: 지도·물리 조건·장면.
- `library/s<ID>/k<K>/`: pose NPY, canonical 좌표, coverage/solver result, 실행 시 validation.
- `transfers/s<ID>/k<K>/report.json`: 세 방법의 상태/수치, source 인증/hash, 센서 대응, u/alpha, 투영 변위, runtime.
- 같은 transfer 폴더의 `*_coverage.npz`: 새 지도 target과 관측 density. 전역 후보 cache가 아니라 선택 배치의 평가 결과.
- 같은 폴더의 `*.html`: 결과 하나씩 회전/확대 가능한 Plotly 3D. `index.html`에서 연결합니다.

이 실행에는 별도의 전체 집계 CSV/PPT/PDF가 생성되지 않았습니다. 원자료는 manifest와 각 report이고,
`index.html`은 3D 링크/상태표입니다. 결과물은 ignored 경로이므로 clone에 자동 포함되지 않습니다.

검증 명령: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_library_expansion.py`.
6개 테스트에 입력/분리 검증, dry-run 무실행, 불가 결과 기록, 축소 후보의 실제 library 생성과
새 지도 direct raycast 연결 검증이 포함됩니다. 축소 테스트는 정식 실험 결과가 아닙니다.

저장 결과 수와 수치 일관성만 다시 확인하는 명령(학습/solve/raycast 없음):

```bash
.venv/bin/python - <<'PY'
import json
from collections import Counter
from pathlib import Path
import numpy as np

root = Path('results/prototype_library_shapes_20261008')
manifest = json.loads((root / 'manifest.json').read_text())
print('run:', manifest['status'], manifest['finished_at'])
print('library maps:', len(manifest['library']))
print('library K results:', sum(len(x['results']) for x in manifest['library']))
print('slots:', Counter(v['status'] for x in manifest['transfers'] for v in x['placements'].values()))
checked = 0
for path in sorted((root / 'transfers').glob('*/k*/report.json')):
    report = json.loads(path.read_text())
    for name, value in report['placements'].items():
        if value['status'] != 'complete':
            continue
        poses = np.load(path.parent / value['positions_npy'], allow_pickle=False)
        assert poses.shape == (value['num_selected'], 5) and np.isfinite(poses).all()
        with np.load(path.parent / f'{name}_coverage.npz', allow_pickle=False) as data:
            n = len(data['targets'])
            covered = int((data['density'] > 0).sum())
            assert n == value['target_count'] and covered == value['covered_count']
            assert abs(100 * covered / n - value['coverage_pct']) < 1e-9
        assert (path.parent / value['view']).is_file()
        checked += 1
print('saved array checks:', checked)
PY
```

현재 기대값: library 4개/24개 K 결과, `complete: 50`, `unavailable: 22`, saved array checks 50.

### 다음 작업과 완료 기준

1. 독립 library 원본은 유지하고, 대응 불가 K별로 설치선 배분과 다른 feasible 배치의 필요성을 분석합니다.
2. 대응 가능한 보조 배치를 추가한다면 설치선 배분 제약·목적값 손실을 별도로 저장합니다.
   기존 그룹당 한 대 구현의 확장 여부부터 검토하며, 보조 배치를 무제약 최적해로 표시하지 않습니다.
3. 비교 대상인 단독 전이 2개와 가능한 보간 각각에서 동일 예산의 fresh local MAPPO를 수행합니다.
   새 target의 전체 후보 cache는 주지 않고 실제 조회 위치와 관측 구성 비용을 계측합니다.
4. online 결과와 프로토콜을 고정한 뒤 동일 target/K의 전체 이산 후보 cache와 MILP 기준값을 별도로 계산합니다.
   top-10 기준선 및 인증 없는 시간 제한 해와 구분합니다.
5. coverage 차이(%p), feasible MILP와의 차이/유한 bound, 총 online 시간, ray/평가 수, offline 구축 비용을 함께 보고합니다.
   보간 불가율도 보고하고 library 구축 비용의 재사용 횟수에 따른 상각을 별도로 분석합니다.

위 항목은 **10월 8일 실행 당시의 후속 목록**입니다. 이후 동일 초기값 LS/MAPPO 72회와
target full-grid 8개 및 archive 검증을 완료했습니다. library 생성 실행 자체에는 그 계산이 없으며,
각 결과를 날짜별 독립 실행으로 구분합니다. 대응 보조 배치·공정한 반복 시간 비교 등은 여전히 미완료입니다.

## 이전 Prototype 연구 경로와 읽는 순서

이 README는 실행 코드와 저장 결과에 근거한 방법·실험 명세입니다. 외부 문헌 검토나 논문 완성본이
아니며, 미구현 기능과 실험으로 입증하지 못한 주장은 아래에 따로 표시합니다.
2026-10-09 갱신에서는 새 학습/대규모 solve를 실행하지 않고 10월 8일 완료 산출물을 대조했습니다.

| 확인할 내용 | 위치 |
|---|---|
| 최신 library 24개와 전이/보간 전체 결과 | 위 `라이브러리 형상 확장 (2026-10-08)` |
| 최신 3D 및 보간 불가 사유 | [library expansion index](results/prototype_library_shapes_20261008/index.html) |
| 설치 구간 비율 보간 수식 | `라이브러리 형상 확장`, `순차 Runtime 비교 경로` |
| 기존 거리 보간·국소 MAPPO 수식 | `폭 변화 실험 방법론 (2026-10-03)` |
| 15개 새 폭 실험의 프로토콜·전체 수치·재현 명령 | `미사용 도로 폭 15개 확장 실험` |
| 15개 폭 실험의 그래프와 3D | [width sweep index](results/prototype_width_sweep_20261003/index.html) |
| 15개 폭 실험 seed 원자료 / 평균·표본 SD | [trials.csv](results/prototype_width_sweep_20261003/trials.csv) / [summary.csv](results/prototype_width_sweep_20261003/summary.csv) |
| 이전 비대칭 4개 맵의 MAPPO·MILP 비교 | [shape comparison](results/prototype_shape_baselines_20261008/method_comparison.html) |
| 10m 개발 파일럿 | `도로 상대 좌표 전이와 MAPPO 국소 보정`, `설치선 배분을 맞춘 보조 Prototype` |
| 공통 물리 모델과 기존 MILP/GREEDY/MAPPO 수식 | 번호 4~10절 |
| 사용자 가중 목적함수와 UI | `배치 Workbench` (최신 prototype 실험과 별개) |
| 과거 실험 | `기존 전체 도로용 MAPPO v3 결과`, 번호 12~15절의 v2 세미나 기록 |

최신 비교에서는 **무가중 union coverage**만 배치 선택 기준으로 사용합니다.
UI의 importance/overlap/count/cost 슬라이더 값은 이 실험에 적용되지 않았습니다.

### 구현 경로 구분

| 경로 | 학습·선택 방식 | 기존 지식 재사용 | 결과 의미 |
|---|---|---|---|
| 기존 Roadside GREEDY/MILP | 공통 이산 후보의 union 최적화 | raycast cache | at-most-K 배치 |
| 기존 Roadside MAPPO v3 | 11-action, scenario/K/trial별 새 학습 | 물리 cache, warm spawn | 탐색/정책 평가/yaw 후처리 중 최고 |
| prototype + local MAPPO | 상대 좌표 전이 후 11-action 국소 새 학습 | **pose·geometry만**, 정책 가중치 미사용 | 초기/탐색/평가 중 최고, yaw 후처리 없음 |
| 최신 library 형상 확장 | source 전체 후보/K별 최적화 후 세 전이 슬롯 평가 | 독립 source pose·geometry | 보간 가능 여부 및 초기 배치 수치, MAPPO 미실행 |
| UI candidate MAPPO | 후보 ID 또는 미설치 선택, 1-step bandit | 같은 장면 visibility | 공통 가중 목적값의 최고 탐색 배치 |

이 prototype 계열의 방법은 **배치 전이 + 새 맵에서의 RL 기반 국소 탐색**입니다.
당시 library 확장은 그 앞단의 데이터 구축·대응 가능성 확인 단계입니다.
정책을 zero-shot으로 재사용하거나 pretrained policy를 fine-tuning하는 구현이 아닙니다.
이후 source 기여 분리 결과를 바탕으로 최신 주 경로는 [source-free geometry 선별](#current-status)로
이동했습니다. 아래 과거 학습 설정은 최신 corridor에 적용되지 않습니다.

## 단순 형상 확장 파일럿 (2026-10-07)

OSM 원본 지도 확장은 미루고, 기존 100m x 100m 파라메트릭 맵 안에서 교차점 위치를 바꾸는
개발 맵 4개를 추가했습니다. 도로 폭 10m, 직각 교차, 인도/건물/LiDAR 모델은 유지합니다.
지도 전체를 함께 평행이동하는 것이 아니라 고정된 지도 경계 안에서 교차점을 옮기므로
각 도로 가지의 길이와 관측 대상 분포가 달라집니다. 곡선/L자/Y자는 이번 범위에 없습니다.

입력 명세: [configs/shape_pilot.json](configs/shape_pilot.json).

| ID | 이름 | 형상 | 교차점 (x,y), m | 동/서/남/북 가지 길이, m |
|---:|---|---|---|---|
| 400001 | t_left | 좌우 비대칭 T자 | (35,50) | 65 / 35 / 50 / 없음 |
| 400002 | t_long_stem | 아래쪽 가지가 긴 T자 | (50,65) | 50 / 50 / 65 / 없음 |
| 400003 | cross_left | 좌우 비대칭 십자 | (30,50) | 70 / 30 / 50 / 50 |
| 400004 | cross_asymmetric | 가로·세로 비대칭 십자 | (40,65) | 60 / 40 / 65 / 35 |

가지 길이는 교차점에서 지도 경계까지의 거리이며 실제 설치 가능 구간 길이와 다릅니다.
원본 scenario 0~9와 `configs/experiment.yaml`을 수정하지 않았습니다.
새 geometry는 `Scenario.raw.geometry.junction_x_m/junction_y_m`으로 명시적으로 활성화합니다.
도로/인도/건물 mesh, target, 설치 판정, branch frame은 공통 `src/environment.py`에 연결되며
MAPPO용 별도 물리 환경을 만들지 않았습니다. geometry 설정은 새 cache hash에 포함됩니다.
기존 설정을 사용하는 맵의 cache hash·기하 기본값은 유지합니다.

### 이번 배치 전이의 의미

기존 7m/14m 중앙 교차로 prototype을 사용합니다. 두 원본의 교차점 위치는 모두 (50,50)입니다.
폭 10m에 대한 보간 계수는 `(10-7)/(14-7)`이고, 보간한 설치 구간 비율 `u`를
새 비대칭 맵의 동일 branch/side 설치 구간에 복원합니다.
따라서 **폭 보간 + 새로운 가지 길이로의 배치 전이**이지, 두 교차점 위치 사이의 보간은 아닙니다.
새 형상 전이에 대한 개발용 stress test이며 unseen 성능 입증이나 최적성 보장은 아닙니다.

각 맵에서 low 단독 전이/high 단독 전이/보간을 직접 평가하고 최고 배치를 선택합니다.
선택만 한 결과, 해당 초기 배치의 local-search 보정, MAPPO 보정을 비교합니다.
local-search와 MAPPO는 동일한 초기 배치에서 독립적으로 시작하며 앞선 보정 결과를 이어받지 않습니다.
수평/수직 설치선을 유지하므로 기존 11-action과 반경 10m 제약을 그대로 사용합니다.

### 실행과 자료

아래 명령은 K=3, MAPPO seed=0/최대 256 joint steps, local-search 최대 128 이웃 평가,
보정당 soft deadline 30초, CPU 순차 worker 12개입니다. **대규모 학습이 아닌 짧은 개발 실행**입니다.
이 pilot에서는 새 맵의 GREEDY/MILP 전체 재최적화를 실행하지 않으므로 optimum gap이나
full optimization 대비 speedup은 TBD입니다. 기존 YAML의 PPO 설정은 유지합니다.

```bash
# 계획만 확인
.venv/bin/python prototype_runtime.py \
  --output results/prototype_shape_plan \
  --shape-config configs/shape_pilot.json \
  --methods select local_search mappo --modes fraction --seeds 0 \
  --steps 256 --evaluations 128 --refine-seconds 30 --gpu cpu

# 실제 개발용 실행. 기존 output이 있으면 새 이름 사용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_runtime.py \
  --output results/prototype_shape_pilot_20261007 \
  --shape-config configs/shape_pilot.json \
  --methods select local_search mappo --modes fraction --seeds 0 \
  --steps 256 --evaluations 128 --refine-seconds 30 \
  --gpu cpu --save-3d --execute
```

`--shape-config`가 있으면 `--scenarios/--widths` 조합 대신 해당 JSON의 맵을 사용합니다.
`--methods ... greedy milp`로 새 맵 전체 후보 baseline을 별도로 추가할 수 있으며 실제 계산량은 커집니다.
결과의 `scenario`는 source library의 parent ID이고, `target_scenario_id`가 실제 새 맵 ID입니다.

- [맵과 3D 결과 목록](results/prototype_shape_pilot_20261007/index.html)
- [맵 형상 4개](results/prototype_shape_pilot_20261007/maps.png)
- [전체 비교 CSV](results/prototype_shape_pilot_20261007/comparison.csv)
- 각 worker: `initial_candidates.json`, `initial.npy`, `deployment.npy`, `coverage.npz`,
  `result.json`, `geometry.json`, `config.yaml`, `deployment.html`.
- MAPPO worker: 별도의 `refinement/`에 학습 로그·최종 정책·최고 배치를 저장합니다.
- `initial_candidates.json`은 세 초기 배치의 pose이고, 각 coverage는 `result.json.selection.alternatives`입니다.

검증 명령: `.venv/bin/python tests/test_shape_pilot.py` (6 tests).
기존 중앙 맵 회귀, geometry별 cache 분리, 좌표 round-trip, shifted 맵 설치 제약,
실제 raycast의 건물 위치 변경, 4개 맵/12개 실행 계획을 검사합니다.

### 개발 실행 결과

2026-10-07 14:08:50~14:10:44 KST에 12/12 worker를 완료했습니다.
MAPPO 4개는 모두 256 steps/1 PPO update를 수행했습니다. 아래 coverage는 초기 배치를 보존한
최고 deployment이며 최종 정책의 단독 평가값이 아닙니다. 단위는 %입니다.

| 맵 | Low 전이 | High 전이 | 보간 | 선택 후 국소 탐색 | 선택 후 MAPPO |
|---|---:|---:|---:|---:|---:|
| t_left | 87.015 | 87.357 | 86.065 | 95.349 | 90.938 |
| t_long_stem | 82.879 | 83.552 | 83.672 | 89.052 | 84.305 |
| cross_left | 70.378 | 75.029 | 73.241 | 76.250 | 75.116 |
| cross_asymmetric | 73.551 | 72.083 | 73.377 | 73.653 | 78.101 |

아래 시간은 입력/전이/초기 배치 선택부터 최종 재평가까지의 CPU 실행 wall time입니다.
프로세스 시작과 최종 3D/report export는 제외하며, 반복 측정 통계는 아닙니다.

| 맵 | 선택만, 초 | 선택+국소 탐색, 초 | 선택+MAPPO, 초 | MAPPO 최종 정책 coverage, % |
|---|---:|---:|---:|---:|
| t_left | 3.18 | 4.05 | 8.78 | 89.747 |
| t_long_stem | 3.13 | 4.10 | 10.05 | 75.305 |
| cross_left | 3.19 | 4.38 | 9.98 | 42.175 |
| cross_asymmetric | 3.37 | 4.51 | 10.26 | 78.101 |

해석: 보간이 초기 배치 선택에서 이긴 것은 4개 중 1개이며, 국소 탐색이 MAPPO보다 높은
deployment coverage를 얻은 것은 4개 중 3개입니다. 비대칭 십자에서는 MAPPO가 더 많이 개선했습니다.
이 짧은 실행은 보정 방법의 형상별 차이를 확인하는 개발 증거입니다. 단일 seed/1 PPO update이므로
MAPPO의 수렴이나 방법 간 일반적인 우열을 주장하지 않습니다. 특히 초기 배치 보존값과
최종 정책 성능의 차이가 크므로 본 실험은 더 긴 학습과 여러 seed를 분리 평가해야 합니다.
이 파일럿 당시에는 새 맵의 GREEDY/MILP baseline이 없었습니다. 후속 4,096-step 실행과
2026-10-08 기준선은 아래에 별도로 기록합니다. 전체 연속 공간 최적성과 학습 수렴 검증은 TBD입니다.

최종 pose 12개를 저장된 density에 의존하지 않고 새 mesh raycasting으로 다시 계산하여
target/density/coverage/설치 제약/반경/학습 로그 검사를 통과했습니다.
검증 기록은 `validation.json`이며 아래 명령으로 재검산할 수 있습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_shape_pilot.py results/prototype_shape_pilot_20261007
```

브라우저에서는 12개 3D 링크, 대표 비대칭 십자의 실제 WebGL 렌더링·카메라 회전,
모바일 결과 목록의 viewport overflow 없음과 JavaScript 오류 없음을 확인했습니다.

### PPO 학습량 확장: 4,096 steps

2026-10-07 17:26:37~17:33:09 KST에 같은 4개 맵의 MAPPO 실행을 완료했습니다.
K=3, seed=0, fraction 전이, 반경 10m, CPU 순차 실행을 유지하고 joint steps만
256에서 4,096으로 늘렸습니다. 시간 제한은 학습 조기 종료를 피하도록 30초에서 600초로
높였으며 4개 모두 조기 종료 없이 완료했습니다. rollout 256, PPO epochs 5는 그대로이므로
업데이트는 실행당 1회에서 16회로 증가했습니다. epoch 수를 늘린 실험은 아닙니다.
같은 초기 pose를 검증했으며, 기존 정책 checkpoint를 이어 학습하지 않고 새로 초기화했습니다.
학습은 전체 visibility cache 대신 on-demand raycasting과 실행 내부 memoization을 사용합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_runtime.py \
  --output results/prototype_shape_ppo4096_20261007 \
  --shape-config configs/shape_pilot.json \
  --methods mappo --modes fraction --seeds 0 \
  --steps 4096 --refine-seconds 600 --gpu cpu --save-3d --execute
```

재실행에는 새 output 경로를 지정합니다. 기본 `configs/experiment.yaml`은 변경하지 않았습니다.
아래 coverage는 %, 증가는 %p입니다. 최고 배치와 마지막 deterministic 정책 평가를 구분합니다.

| 맵 | 256-step 최고 배치 | 4,096-step 최고 배치 | 증가 | 마지막 정책 | online 시간, 초 |
|---|---:|---:|---:|---:|---:|
| t_left | 90.938 | 96.572 | +5.634 | 93.340 | 89.21 |
| t_long_stem | 84.305 | 89.385 | +5.080 | 84.328 | 89.52 |
| cross_left | 75.116 | 81.405 | +6.289 | 80.305 | 95.82 |
| cross_asymmetric | 78.101 | 78.401 | +0.300 | 69.477 | 95.42 |

시간은 입력/전이/초기 선택/학습/평가/정책 저장/최종 재검산을 포함하며 eager import,
worker 시작, 최종 3D export는 제외합니다. 전체 launcher 경과 시간은 약 392초입니다.
기존 국소 탐색은 128회 이웃 평가, MAPPO는 4,096 joint steps로 계산 예산이 다르므로
이 표만으로 계산 효율이나 일반적인 알고리즘 우위를 주장하지 않습니다.

학습량 증가 후 네 맵의 최고 배치는 개선됐지만, 비대칭 십자의 마지막 정책은
기존 78.101%에서 69.477%로 낮아졌습니다. 최고 탐색 배치의 개선은 정책 수렴을 뜻하지 않습니다.
여전히 단일 seed의 개발 실험이며 다중 seed와 동일 시간 예산 검증은 TBD입니다.
새 맵에서 처음부터 계산한 GREEDY/MILP 비교는 아래 2026-10-08 절에 추가했습니다.

- [4개 맵의 3D 결과 목록](results/prototype_shape_ppo4096_20261007/index.html)
- [결과 및 runtime CSV](results/prototype_shape_ppo4096_20261007/comparison.csv)
- 각 worker의 `result.json`: 실제 steps/업데이트 수, PPO 진단값, 최고 배치/최종 정책, runtime.
- 각 `refinement/`: `train_log.npz`, `policy.pt`, 최고 배치/최종 정책/평가 최고 NPY.
- `manifest.json` 및 `inputs/`: 실제 명령, 설정, source/prototype hash와 입력 snapshot.

4개 저장 배치의 direct raycast 재검산, 설치/반경 제약, target density 및 로그 유한값을
검증했습니다. 업데이트 16회, steps 4,096, 동일 초기 pose, PPO 진단값 유한성도 확인했습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_shape_pilot.py results/prototype_shape_ppo4096_20261007
```

### 추가 형상 GREEDY/MILP 기준선 (2026-10-08)

같은 4개 맵, K=3에서 GREEDY 4개와 MILP 4개를 완료했습니다.
지도/target/LiDAR 조건은 기존 국소 보정과 같고, baseline은 전역 설치 후보를 생성하여
처음부터 선택합니다. 위치별 top-10 자세 pruning과 coverage-mask dedup은 유지합니다.
따라서 이는 **pruned-candidate baseline**이며 모든 자세/연속 공간의 최적해가 아닙니다.
MAPPO는 초기 배치 주변 반경 10m에서 자세를 탐색하므로 feasible space도 동일하지 않습니다.

실제 원 실행 명령:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_runtime.py \
  --output results/prototype_shape_baselines_20261008 \
  --shape-config configs/shape_pilot.json --methods greedy milp --modes fraction \
  --k 3 --solver-seconds 300 --gap 0.05 --topm 10 --memory-gb 26 --gpu cpu \
  --save-3d --execute
```

작업별 새 process/cache를 사용하고 CPU에서 순차 실행했습니다. Gurobi 13.0.1,
1 thread, solver 제한 300초, relative gap 5%, soft memory limit 26GB입니다.
원 실행은 5개 완료 후 중단됐고, 같은 입력/source hash를 확인한 뒤 나머지 3개만 재실행했습니다.
중단된 계산은 solver checkpoint에서 이어진 것이 아니라 처음부터 재시도했습니다.
완료 5개는 보존했고 중단 당시 파일/로그/manifest는 `interrupted_attempts/`에 있습니다.
`resume_history`에 재실행 명령과 helper hash를 기록했습니다.

```bash
# 해당 로컬 결과 폴더에 보존된 재실행 helper. 완료 상태에서는 재계산하지 않음.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u \
  results/prototype_shape_baselines_20261008/resume_remaining.py --execute
```

#### Coverage 비교

단위 %. MAPPO는 seed=0, 4,096-step 실행의 최고 보존 배치이며 최종 정책 성능이 아닙니다.
국소 탐색은 앞선 128 이웃 평가 실행입니다. 모든 최종 baseline은 실제 3대를 선택했습니다.

| 맵 | 전이 후보 중 최고 | 국소 탐색 | MAPPO 4,096 | GREEDY top-10 | MILP top-10 |
|---|---:|---:|---:|---:|---:|
| t_left | 87.357 | 95.349 | 96.572 | 92.516 | 99.905 |
| t_long_stem | 83.672 | 89.052 | 89.385 | 89.833 | 99.845 |
| cross_left | 75.029 | 76.250 | 81.405 | 86.652 | 92.975 |
| cross_asymmetric | 73.551 | 73.653 | 78.401 | 90.654 | 97.660 |

#### 계산시간과 Solver 인증

전체 online wall time은 입력/geometry/전체 visibility/후보 생성/선택 또는 solve/최종 재검산을
포함하며 eager import, worker 시작, 최종 그림 export는 제외합니다. baseline cache 쓰기도 포함됩니다.
중단된 첫 시도의 비용과 사용자 대기 시간은 아래 성공 실행 시간에 합산하지 않았습니다.
기존 MAPPO는 다른 날 실행한 단일 측정이며, 반복 측정/동일 시간 예산의 speedup 증명은 아닙니다.

| 맵 | GREEDY 전체, 초 | MILP 전체, 초 | MILP solver만, 초 | MAPPO 전체, 초 |
|---|---:|---:|---:|---:|
| t_left | 45.35 | 200.48 | 148.94 | 89.21 |
| t_long_stem | 49.46 | 150.05 | 93.10 | 89.52 |
| cross_left | 63.56 | 344.71 | 270.63 | 95.82 |
| cross_asymmetric | 61.85 | 320.53 | 245.06 | 95.42 |

전체 시간이 300초를 넘을 수 있는 이유는 제한이 solver에만 적용되기 때문입니다.
아래 objective/bound는 관측 target 개수 단위이며, 최대화 문제에서 best bound는 최적값의 **상한**입니다.

| 맵 | Gurobi status | Objective | Best bound | Relative gap |
|---|---:|---:|---:|---:|
| t_left | 2 | 15,765 | 15,765 | 0% |
| t_long_stem | 2 | 17,373 | 17,373 | 0% |
| cross_left | 2 | 19,190 | 19,392 | 1.053% |
| cross_asymmetric | 2 | 20,157 | 20,157 | 0% |

3개는 주어진 후보 집합에서 solver-reported gap 0%이며, cross_left는 설정 tolerance 내 종료입니다.
status 2만으로 모든 결과를 gap 0% 최적해라고 해석하지 않습니다.

현재 MAPPO 최고 배치는 4개 모두 MILP보다 낮고, GREEDY보다 높은 것은 t_left 한 개입니다.
MAPPO는 이 실행에서 MILP보다 시간은 짧지만 GREEDY보다 오래 걸렸습니다.
따라서 단순히 학습량을 늘린 것으로 near-optimal 전이나 효율 우위를 입증하지 못했습니다.
다음 검증은 동일 시간 예산의 국소 탐색/MAPPO, 여러 seed 및 초기 배치/국소 반경의 영향입니다.
오프라인 library 구축 비용의 amortization도 별도로 계산해야 합니다.

#### 자료와 재검산

- [통합 비교 그래프·표·3D 링크](results/prototype_shape_baselines_20261008/method_comparison.html)
- [24개 비교 원자료 CSV](results/prototype_shape_baselines_20261008/method_comparison.csv)
- [그래프 PNG](results/prototype_shape_baselines_20261008/method_comparison.png) / [PDF](results/prototype_shape_baselines_20261008/method_comparison.pdf)
- [기준선 8개 3D 목록](results/prototype_shape_baselines_20261008/index.html)
- 각 worker: `result.json`, pose/density, `config.yaml`, `geometry.json`, `deployment.html`; MILP는 `gurobi.log` 추가.
- `validation.json`: 8개 배치의 새 direct raycast, target/density/coverage/설치 제약 재검산 통과.
- `comparison_provenance.json`: 집계 명령, report source hash, 입력 result hash.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_shape_pilot.py results/prototype_shape_baselines_20261008

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python shape_baseline_report.py \
  --roots results/prototype_shape_pilot_20261007 \
          results/prototype_shape_ppo4096_20261007 \
          results/prototype_shape_baselines_20261008 \
  --output results/prototype_shape_baselines_20261008
```

집계는 완료/검증 상태, 중복, 동일 geometry/target 순서/LiDAR/K와 각 맵의 6개 방법을 검사합니다.
결과 경로는 git ignored이므로 clone만으로 이 자료가 포함되지는 않습니다.

## 순차 Runtime 비교 경로 (2026-10-06)

기존 결과·실행 기본값을 보존하고 `prototype_runtime.py`를 추가했습니다.
아래는 **새 구현 및 검증 경로**이며, 기존 15개 맵 실험의 시간을 소급해서
정정한 결과가 아닙니다. 새 방식의 정식 성능·속도 비교는 **TBD**입니다.

### 비교하는 방법

1. `distance`: 기존 교차로 경계 기준 거리 보간.
2. `fraction`: 동일 branch/side 설치 구간 안에서의 비율 보간.
3. 각 방식에서 `transfer_low`, `transfer_high`, `interpolation`을 직접 raycasting으로 평가합니다.
4. 최고 coverage의 초기 배치 **하나만** 선택합니다. 동률은 low → high → interpolation 순서입니다.
5. `select`는 여기서 종료하고, `local_search` 또는 `mappo`는 선택한 배치만 보정합니다.
6. 별도 GREEDY/MILP worker는 새 지도에서 전체 visibility/candidate를 생성해 처음부터 풉니다.

초기 세 배치의 평가 비용도 online 시간에 포함됩니다. 세 배치를 모두 학습한 뒤 최고값만
보고하는 oracle 비교가 아닙니다. 불가능한 보간은 unavailable로 남기고 단독 전이만 비교합니다.
단독 전이 자체도 불가능한 경우는 실패 처리하며 임의의 위치로 대체하지 않습니다.

### 구간 비율 보간

`prototype_segments.py`가 공통 `MapGeometry.is_installable()`의 0.5m grid를 순회하여
branch/side별 종방향 설치 구간 `[s_min, s_max]`를 추출합니다. coverage는 구간 추출·센서 대응에
사용하지 않습니다. 현재는 branch/side마다 종방향 구간 하나인 파라메트릭 도로만 지원하며,
분리된 구간이나 퇴화 구간은 오류로 처리합니다. 인도 안의 두 평행 grid 줄은 같은 branch/side로
묶고 횡방향 편차 `e`를 별도 보존합니다. 임의 OSM 도로망의 자동 대응 기능은 아닙니다.

```text
u_i = (s_i - s_min) / (s_max - s_min)
alpha = (width_target - width_low) / (width_high - width_low)
u_target = (1-alpha) * u_low + alpha * u_high
s_target = s_min_target + u_target * (s_max_target - s_min_target)
d_target = side_sign * (target_sidewalk_center_offset + e_target)
```

`e`, 높이, pitch는 선형 보간하고 상대 yaw는 최단 원호로 보간합니다.
기존 template reflection + Hungarian matching을 사용하되 low 배치를 high 구간의 비율 좌표로
옮긴 뒤 같은 branch/side의 센서끼리 대응시킵니다. 따라서 이번 ablation에는 대응 cost에 사용하는
좌표 변경도 포함됩니다. 매칭 cost는 기존 기하·자세 거리 heuristic이며 최적성 증명이 아닙니다.
단독 전이도 같은 `u` 변환을 사용합니다. 마지막에 기존 XYZ/yaw/pitch 후보 격자로 투영하고
설치 제약·완전 중복 pose를 검사합니다. source/target 구간, u, 대응표, 투영 변위를 JSON에 저장합니다.

`0 <= u <= 1`이면 복원된 종방향 좌표가 구간 안에 있다는 성질만 보장됩니다.
장애물 가림·FOV·센서 간 중복 때문에 coverage 또는 최적 배치의 선형 보간 성능은 보장되지 않습니다.
직선 도로는 폭만 달라질 때 종방향 구간 길이가 같으므로 두 방식의 결과가 같을 수 있습니다.

### 국소 탐색과 시간 예산

`src/methods/local_search.py`는 한 번에 한 센서의 행동을 바꿔 가장 큰 coverage 증가를 선택하는
deterministic best-improvement 탐색입니다. 기존 `LocalPPOAgent`의 11-action 중 stay를 제외한
행동, action mask, yaw/pitch clamping, 초기 XY 기준 반경 제한을 그대로 사용합니다.
동시 다중 센서 이동은 없으며 평가 상한/시간 제한/국소 개선 없음에서 종료합니다.
관측 state 구성 비용도 포함되므로 최적화된 전용 local-search 구현의 하한 시간은 아닙니다.

MAPPO는 선택한 배치에서 정책을 새로 학습하며 기존 reward/PPO를 유지합니다.
두 방법 모두 초기 best를 보존하고 최종 배치를 공통 direct evaluator로 재검산합니다.
`--refine-seconds`는 초기화부터 세는 **soft deadline**입니다. 진행 중인 평가, PPO update,
잔여 rollout 처리, 최종 policy 평가·저장 때문에 실제 시간이 초과될 수 있습니다.
따라서 같은 설정 시간만으로 동일 계산 예산이라고 주장하지 않고 실제 wall time도 보고해야 합니다.
`--steps`는 MAPPO 최대 joint steps, `--evaluations`는 local-search 최대 이웃 평가 수입니다.

### 계측 범위와 산출물

- 각 조합은 새 Python subprocess에서 **순차** 실행합니다. 지정한 GPU 하나 또는 CPU를 사용합니다.
- 입력/geometry/전이, 초기 배치 평가, 전체 후보 준비, 선택/solve, 보정, 최종 평가를 분리 기록합니다.
- online wall time에는 baseline cache 저장과 MAPPO checkpoint 저장도 포함됩니다.
  worker 시작·최상위 import와 최종 보고서/3D export는 제외하고, 구간 내부 lazy import는 포함합니다.
- 전체 worker wall time도 따로 저장합니다. 외부 작업 부하는 제어하지 않으며 공정한 비교에는 전용 실행 시간이 필요합니다.
- `rays_submitted`는 실제 raycast 호출에 제출한 ray 수입니다. 실패 후 재시도도 중복 계수합니다.
  raycast 호출, LOS query, 직접 pose/캐시 pose/후보 pose 평가, reward 평가, joint step 수를 구분합니다.
  이는 FLOP 수가 아니며 각 종류의 평가 비용이 동일하다는 뜻도 아닙니다.
- 최초 설치·import를 포함한 process-lifetime RSS peak와 PyTorch allocator peak를 기록합니다.
  GPU peak는 Open3D/driver 등 전체 GPU 메모리가 아닙니다.
- `cpu_process_sec`는 계측 구간의 process CPU 시간이며 GPU 연산 시간 또는 wall time과 같지 않습니다.
- 알고리즘 cache는 조합마다 새로 만듭니다. 같은 run 안의 on-demand memoization은 허용합니다.
  OS filesystem cache는 비우지 않으므로 하드웨어 수준의 완전한 cold-cache 측정은 아닙니다.
- `quality_trace`에는 평가가 끝난 실제 시각의 배치 coverage를 남깁니다.
  MILP 중간 incumbent callback은 미구현이며 warm start와 solver 반환 시점만 기록합니다.
- `manifest.json`, source/prototype snapshot·hash, `packages.txt`, 조합별 log/result/config/geometry,
  pose NPY와 density NPZ, 전체 `comparison.csv`를 저장합니다. 실패를 coverage 0으로 집계하지 않습니다.
- `--save-3d`를 추가하면 각 조합에 회전 가능한 `deployment.html`을 생성합니다.

### 실행 명령

계획만 확인합니다. raycasting/학습/solve와 output 저장을 하지 않습니다.

```bash
.venv/bin/python prototype_runtime.py \
  --output results/prototype_runtime_plan \
  --scenarios 0 1 2 --widths 10 --methods select local_search
```

아래는 **실제 계산** 명령입니다. 새 root만 허용하며 기존 결과는 덮어쓰지 않습니다.
K=3, 3개 topology, 두 좌표 방식, 세 초기 배치 중 선택, MAPPO 3 seeds,
deterministic 방법 각 1회와 별도 full-build GREEDY/MILP까지 총 36개 순차 작업입니다.

```bash
.venv/bin/python prototype_runtime.py \
  --output results/prototype_runtime_official_v1 \
  --scenarios 0 1 2 --widths 10 --modes distance fraction \
  --methods select local_search mappo greedy milp \
  --seeds 0 1 2 --steps 2000 --evaluations 2000 --refine-seconds 60 \
  --solver-seconds 300 --gap 0.05 --topm 10 --memory-gb 26 \
  --gpu 0 --save-3d --execute
```

`--repeats`는 독립 process의 timing 반복이고 새로운 MAPPO seed를 자동 생성하지 않습니다.
`--topm 0`은 전체 orientation이며 메모리/시간 부담이 큽니다. 기본은 top-10이므로
baseline을 full-orientation global optimum이라고 부르면 안 됩니다.
이번 runner는 Gurobi 오류를 CBC로 조용히 바꾸지 않습니다.
off-line library 구축 비용의 amortization, warm-cache 별도 실험, unseen 평가셋, 동일 품질 도달 시간의
통계 분석은 후속 과제입니다. 기존 8/9/10/11/12/13m 결과를 보고 방법을 고르면 해당 맵들은 개발셋입니다.

경량 검증:

```bash
.venv/bin/python tests/test_prototype_runtime.py
```

추가 테스트 8개에서 구간 좌표 round-trip/투영, 실제 ray 수와 cache 재사용 계수,
초기 배치 선택, local-search 제약/재평가/결정성, 8-step PPO 및 초기화 중 deadline 도달,
실행 보호장치와 작은 GREEDY/MILP wrapper를 확인했습니다.
실제 source library를 사용한 경량 worker 6개는 다음 명령으로 완료했습니다.

```bash
.venv/bin/python prototype_runtime.py \
  --output results/prototype_runtime_smoke_20261006 \
  --scenarios 0 --widths 10 --methods select local_search mappo \
  --seeds 0 --steps 8 --evaluations 8 --refine-seconds 15 --gpu cpu --execute
```

이는 s0의 두 좌표 방식 × 3개 방법 저장/계측 검증입니다. 초기 coverage를 유지하는지 확인한
짧은 실행이지, 새 보간의 성능 개선이나 MAPPO 수렴·속도 우위 실험이 아닙니다.
실행 당시 source는 해당 폴더의 snapshot/hash로 확인하며 이후 추가한 hardware 필드는 이 기록에 없습니다.
hardware·CPU 시간 필드와 worker 설정 고정 검사, HTML 저장은 아래 별도 실행으로 확인했습니다.

```bash
.venv/bin/python prototype_runtime.py \
  --output results/prototype_runtime_guard_check_20261006 \
  --scenarios 0 --widths 10 --modes fraction --methods select \
  --gpu cpu --save-3d --execute
```

예제 3D: [선택된 초기 배치](results/prototype_runtime_guard_check_20261006/s0_w10_fraction_select_r0/deployment.html).
T/C의 기본 high prototype은 기존 low 설치선 배분에 맞춘 보조 library입니다.
무제약으로 독립 최적화한 high prototype과 동일하다고 해석하지 않습니다.

## 폭 변화 실험 방법론 (2026-10-03)

이 절은 중앙 교차로 7m/14m source와 15개 폭 변화 실험의 거리 기반 보간 명세입니다.
10월 7~8일 형상 실험의 구간 비율 보간 및 410xxx/420xxx 신규 library 실행과 구분합니다.
아래 K=3, 중심 (50,50), 2,000-step 설정은 이 절의 실험 조건이지 모든 경로의 현재 기본값이 아닙니다.

### 전체 처리 흐름과 파일 책임

```text
OSM 유래 유형 메타데이터 (실제 polyline 아님)
  -> MapGeometry: 직선 / T자 / 십자, 폭 7m·14m
  -> 전체 이산 XYZ/yaw/pitch 후보 + offline LOS cache
  -> GREEDY warm start + 제한시간 MILP
  -> K별 pose, geometry, solver 인증, 설정/hash 저장
  -> T/십자: 필요하면 설치선 배분을 고정한 14m 보조 prototype 생성
  -> 고정 library + 새 폭 8/9/11/12/13m
  -> 7m 단독 전이 / 14m 단독 전이 / 대응 센서의 상대 좌표 보간
  -> 새 지도의 설치 가능 격자로 투영 + 제약 검사
  -> 직접 raycast 평가
  -> 초기 배치마다 fresh local MAPPO (3 seeds)
  -> 직접 재평가, 새 맵 GREEDY/MILP 기준선, CSV·3D·검산
```

| 파일 | 이 폭 변화 실험에서 담당하는 기능 |
|---|---|
| `prototype_offline.py` | 폭 변형 시나리오, geometry descriptor, canonical 좌표, 전체 후보 MILP library |
| `prototype_compatible.py` | 7m와 설치선별 센서 수를 맞춘 14m 보조 library |
| `prototype_junctions.py` | 원래 T/십자 offline·10m 전이 파일럿 orchestration |
| `prototype_transfer.py` | 원본 로드, 센서 대응, 세 보간 모드, 투영, 직접 평가, 3D export |
| `prototype_refine.py` | 시나리오당 6개 배치 슬롯, 설정/prototype 사본, 새 국소 학습, 보고서 |
| `prototype_benchmark.py` | 15맵·paired seeds·기준선 실행, 완전성 검사, 집계/그래프 |
| `src/methods/local_mappo.py` | on-demand visibility, 초기 pose 주입, 반경 mask, 국소 PPO loop |
| `src/methods/mappo.py` | 공유 actor/critic, agent state/action/reward, time-agent GAE/PPO |
| `src/core/visibility.py` | 공통 각도 변환을 사용하는 explicit pose 직접 Open3D raycast |
| `src/roadside/candidate_generator.py` | XYZ/yaw/pitch 후보, top-M, mask dedup, 그룹별 후보 제한 |
| `src/environment.py` | 공통 물리 환경·설치 제약·target·센서·cache 및 evaluator |
| `tests/validate_prototype_refinement.py` | 저장 prototype으로 전이 재생성, 학습 배치/정책/반경 검산 |
| `tests/validate_width_benchmark.py` | 전체 맵·seed·기준선의 direct density, frozen hash, CSV/SD 검산 |

### 문제 정의와 최적성 범위

환경 `G`, target 집합 `T={q_j}_{j=1..N}`, pose `p_i=(x,y,z,yaw,pitch)`에 대해
`a_ij(G)`는 거리·FOV·가림 조건을 모두 만족하면 1입니다. 공통 물리 수식은 번호 6~7절에 있습니다.

```text
d_j(P) = sum_i a_ij(G)
c_j(P) = 1[d_j(P) >= 1]
C(P;G) = 100/N * sum_j c_j(P)
O(P;G) = 100/N * sum_j 1[d_j(P) >= 2]
U(P;G) = N - sum_j c_j(P)
```

최적화 목적은 `max C(P;G)`이며 중복은 coverage에 한 번만 포함됩니다.
O는 중복 target 비율로, `sum_j max(d_j-1,0)`이라는 초과 관측 횟수와 다릅니다.
target은 도로 XY cell마다 높이 1/1.5/2m의 표본입니다. 3차원 표본 비율이지 검출률·연속 면적·사각지대 크기가 아닙니다.

일반 offline MILP는 이산 후보 `P_grid(G)`에서 `|P|<=K`를 풉니다.
이 paired 전이 실험은 두 원본의 **실제 선택 수가 모두 K=3**이어야 하며 국소 학습도 세 센서를 유지합니다.
GREEDY/MILP 기준선은 여전히 at-most-K입니다. 센서 제거 action이나 K 자동 결정은 없습니다.
전역 연속 공간의 최적성, 모든 K의 최적성, 일정 coverage를 위한 최소 K는 이번 결과에 포함되지 않습니다.

일반 offline 후보는 top-M을 해제하지만 2m XY bin, z=3..6m, yaw 5도, pitch 2.5도의
**이산 격자**입니다. coverage mask가 완전히 같은 후보는 제거됩니다. 순수 union/at-most-K에서는
목적값을 보존하지만 기하학적으로 다른 동가 배치가 library에서 사라질 수 있어 보간에는 영향을 줄 수 있습니다.

MILP 수식은 `max sum(y_j)`, `y_j<=sum(a_ij*x_i)`, `sum(x_i)<=K`, binary `x_i,y_j`입니다.
새 baseline은 단일 coverage 상위 10 orientation만 남깁니다. top-10이 전체 orientation의 최적해를 보존한다는 보장은 없습니다.
Gurobi는 최대화 문제에서 incumbent가 **하한**, best bound가 **상한**입니다.
유한한 bound가 없으면 gap/인증을 알 수 없으며, 그것만으로 첫 LP 미완료라는 원인을 확정하지 않습니다.
`certified()`는 status 2와 목적값·bound·재평가 covered_count의 일치(1e-5 tolerance)를 모두 요구합니다.
시간 제한 best-known을 `optimal prototype`이라고 명명하지 않습니다.

### 실제 사용한 고정 Library

아래 폴더는 모두 `results/` 아래이며 현재 실험은 각 폴더의 `k3/result.json`과 `poses.npy`를 사용했습니다.
coverage는 **각 원본 맵**에서의 값이지 새 폭의 성능이 아닙니다.

| topology·폭 | 원본 폴더 | coverage % | 인증 |
|---|---|---:|---|
| 직선 7m | `prototype_s0_k3_90gb_20261002` | 99.875 | best_known |
| 직선 14m | `prototype_straight_w14_k3_20261002` | 99.872 | best_known |
| T자 7m | `prototype_junctions_20261002/s1_w7` | 90.613 | best_known |
| T자 14m | `prototype_compatible_20261002/s1_w14` | 92.321 | constrained_best_known |
| 십자 7m | `prototype_junctions_20261002/s2_w7` | 79.378 | best_known |
| 십자 14m | `prototype_compatible_20261002/s2_w14` | 85.759 | constrained_best_known |

T/십자의 14m는 원래 독립 최적화 배치가 아니라 7m의 `(branch,side)`별 센서 수를 강제한 보조 배치입니다.
추가 제약은 `sum_{i in group g} x_i = n_g`이며 현재 각 요구 그룹은 1대입니다.
그룹 순서의 모든 순열에 GREEDY를 적용한 최고 union을 warm start로 줍니다.
동일 mask 제거도 그룹 내부에서만 합니다. 300초 MILP가 warm start를 개선하지 못한 결과도 명시적으로 저장합니다.
`14m 단독 전이`는 온라인에서 14m 하나만 사용한다는 뜻이지 offline 생성까지 7m와 독립이라는 뜻은 아닙니다.

library는 visibility cache와 다릅니다. cache는 LOS 조회표, library는 geometry·pose·선택 조건·인증을 묶은 저장물입니다.
`geometry.json`에는 topology, 폭, 차로 수, 지도/건물/인도/분리대, branch heading/endpoints,
설치선/허용오차, 좌표계, 후보/target 조건, provenance를 저장합니다.
`canonical_poses.json`은 pose의 도로 상대 표현이고 `result.json`은 K·coverage·solver·cache/config hash를 담습니다.
센서 설정은 해당 `config.yaml`이 기준입니다. 새맵 descriptors를 이용한 자동 최근접 library 검색은 없고
현재 원본 두 개는 코드/CLI에서 지정합니다. 곡률·교차각·비대칭 차로 폭 등을 학습해 선택하는 기능은 TBD입니다.

변형 맵은 `width_scenario()`가 원래 scenario의 topology와 차로 수를 유지하고 `road_width`만 바꿉니다.
7m 원본은 원래 ID를 쓰고, 변형 ID는 `100000 + parent_id*100000 + round(width_m*1000)`입니다.
예를 들어 s0의 폭 8m는 실제 scenario ID 108000입니다. 표의 s0/s1/s2는 **부모 topology ID**이며
원본 폭 7m의 평가를 뜻하지 않습니다. `provenance=synthetic_width_variant`, `changed_fields=[road_width]`로 기록합니다.

### 도로 상대 좌표 정의

이 거리 기반 경로는 `prepare_relative_placements()`입니다. 기본 지도는 100m x 100m,
교차로 중심 `o=(50,50)`이고 각 branch의 바깥 방향 heading을 `theta_b`로 둡니다.
동/east=0도, 서/west=180도, 북/north=90도, 남/south=270도이며 T자는 북 branch가 없습니다.
branch는 기존 `road_normal_yaw_deg()`로 정하고 교차로 경계의 동률은 기존 가로 도로 우선 규칙을 따릅니다.

```text
t_b = (cos(theta_b), sin(theta_b))
n_b = (-sin(theta_b), cos(theta_b))
s = (xy-o) dot t_b
d = (xy-o) dot n_b
side = sign(d)                 # branch 진행 방향의 left/right
a(w) = w/2                    # 교차로 경계의 종방향 anchor
q = s-a(w)                    # 경계로부터의 종방향 거리
h(w) = w/2 + sidewalk_width/2  # 인도 중심 설치선 offset; sidewalk_width=5m
e = abs(d)-h(w)                # 설치선에서 벗어난 작은 횡방향 offset
psi = wrap_180(yaw-theta_b)    # branch heading 기준 상대 yaw
```

직선은 `theta=0`, `s=x-50`, `d=y-50`, `a(w)=0`, side는 north/south로 저장합니다.
`q`는 정규화된 길이 비율이 아니라 **metres**입니다. 현재 map length와 branch heading은 변하지 않습니다.
지금 보간하는 pose 정보는 `(branch,side,q,e,z,psi,pitch)`입니다.
여기서 `psi`는 branch heading 기준이고, 행동 yaw 허용범위는 **도로 normal** 기준이므로 서로 혼동하지 않습니다.

새 폭 `w*`에 단독 전이하는 식은 다음과 같습니다.

```text
s* = q + a(w*)
d* = side * (h(w*) + e)
xy* = o + s* t_b + d* n_b
yaw* = wrap_360(theta_b + psi)
z*, pitch* = 원본 z, pitch
```

즉 임의의 XY를 학습으로 예측하는 것이 아니라, 종방향 경계 거리와 인도 중심선 offset을 보존하여
새 폭의 설치선으로 옮깁니다. **새 설치선은 새 MapGeometry에서 생성**됩니다.
새 설치선을 원본 두 선의 평균으로 대신하거나 도로 밖의 임의 공간에 센서를 놓는 방식이 아닙니다.

### 센서 대응과 보간 수식

1. 두 원본과 목표의 topology·센서 설정·폭 이외 geometry 조건을 확인합니다. 실제 센서 수도 같아야 합니다.
2. 7m 배치를 14m 상대 좌표계로 옮겨 정렬 비용을 계산합니다.
3. 대칭 반사를 검토합니다. T자는 좌우 반사만, 직선/십자는 좌우·상하 조합을 허용합니다. 90도 회전 탐색은 없습니다.
4. 같은 branch/설치선 쌍만 대응 가능하도록 한 뒤 Hungarian assignment로 센서 순열을 정합니다.
5. 최소 합계 비용의 반사와 순열을 채택합니다. **새 맵 coverage는 이 대응 선택에 사용하지 않습니다.**

대응 비용은 정렬된 공통 14m 좌표계에서 다음과 같습니다.

```text
D_ij = (dx/100)^2 + (dy/100)^2 + (dz/6)^2
     + (wrap_180(yaw_j-yaw_i)/180)^2 + (delta_pitch/90)^2
```

단위별 정규화 상수는 현재 heuristic이며 데이터로 학습한 metric이 아닙니다.
다른 branch/side는 큰 비용을 부여하고, 최종 assignment에 그런 쌍이 있으면 해당 반사를 거부합니다.
어느 반사에서도 대응되지 않으면 보간은 `unavailable`입니다. 이를 단독 전이 결과로 위장하지 않습니다.

원본 폭 `w0=7`, `w1=14`, 목표 폭 `7<w*<14`일 때:

```text
alpha = (w*-w0)/(w1-w0)
v* = (1-alpha)*v0 + alpha*v1,  v in {q,e,z,pitch}
psi* = psi0 + alpha*wrap_180(psi1-psi0)
```

원형 yaw는 예를 들어 179도와 -179도를 0도로 평균내지 않고 짧은 회전 방향을 따릅니다.
branch/side는 이산값이라 보간하지 않습니다. 보간한 상대 좌표를 위 복원식으로 목표 도로에 배치합니다.
현재는 1차원 **도로 폭 보간**이며 다변량 geometry interpolation이나 범위 밖 extrapolation은 아닙니다.

### 설치 격자로의 투영과 모드 차이

복원된 pose는 연속값이므로 `project_poses()`가 목표 맵의 같은 설치선 XYZ 후보에서
`argmin ||xyz_candidate-xyz_raw||^2`로 최근접점을 고릅니다. 해당 위치의 허용 yaw 중 원형 거리가
가장 가까운 값, pitch 격자 중 가장 가까운 값을 선택합니다. coverage로 투영 후보를 선택하지 않습니다.
설치선은 중심선 ±0.25m, XYZ 후보는 0.5m cell 중심을 순회해 2m XY bin당 첫 위치와 z=3/4/5/6m를 조합합니다.
이것은 인도 전체 면적을 자유롭게 탐색하는 것이 아닙니다.

최종 위치·높이·yaw·pitch 격자와 설치 가능 여부를 검사하고 동일한 완전 pose가 생기면 실패합니다.
이 road-relative 투영은 센서별 최근접 선택이며, 충돌 시 다른 점으로 자동 재배정하지 않습니다.
동일 XY에 다른 높이/자세의 센서까지 금지하는 pole occupancy 제약은 없습니다.

| 모드 | 보간 공간 | 대응/투영 | 사용 범위 |
|---|---|---|---|
| `strict` | 폭에 맞게 옮긴 세계 좌표 | 동일 설치선 | 과거 10m 파일럿 |
| `geometry` | 폭에 맞게 옮긴 세계 좌표 | 대응의 branch 제한 해제, 전체 목표 XYZ로 Hungarian 충돌 회피 투영 | 과거 대안 실험 |
| `road_relative` | 경계 anchor·설치선 offset·상대 자세 | 동일 branch/side, 최근접 투영 | **10월 3일 15맵 sweep** |

`prototype_transfer.py`의 기본 mode는 과거 호환용 strict입니다. `prototype_refine.py`와
`prototype_benchmark.py`는 road-relative 함수를 직접 사용합니다. 서로 다른 mode의 점수를 섞지 않습니다.

### 국소 MAPPO의 정확한 역할

각 초기 배치에서 K=3 agent와 actor/critic/optimizer를 새로 생성합니다. 원본 정책 파일을 읽지 않습니다.
agent i의 XY는 초기 위치 `p_i^0`에 대해 `||xy_i-xy_i^0||_2<=10m`라는 원 안으로 제한합니다.
설치 영역과 기존 yaw/pitch/z action mask도 동시에 적용합니다.
stay, x±2m, y±2m, yaw±5도, pitch±2.5도, z±1m의 11-action입니다.
2m 이동은 모든 0.5m 설치선을 연결하지 못하며, 분리된 선/도로 반대편으로 순간 이동하는 기능은 없습니다.
초기 투영 후보 집합과 이후 action으로 도달 가능한 pose 집합도 동일하다고 가정하지 않습니다.

state는 61차원(5 pose + 7x7 local LOS count map + 7 global hints), K=3 critic 입력은 183차원입니다.
local map은 현재 높이에서 1m 간격의 주변 XYZ를 질의하며 FOV 적용 전 LOS target 수를 씁니다.
정규화 분모는 **그 맵의 전체 target 수 N**로, 기존 full-cache MAPPO의 최대 visibility count와 다릅니다.
actor는 local-map CNN(16/32 channels)과 256/128 MLP, critic은 global state의 256/128 MLP와
agent별 value 3개를 출력합니다. 번호 10절의 PPO 구현을 재사용하며 순수 local-only 정책은 아닙니다.

물리 환경은 기존 MapGeometry와 각도 변환을 사용합니다. CPU Open3D `MeshVisibility`가 요청된
XYZ의 range/LOS만 계산하고, 학습의 Torch 경로가 yaw/pitch FOV를 적용합니다.
질의한 XYZ는 해당 run의 메모리에서만 재사용합니다. 관측 local map에 필요한 위치도 질의 대상입니다.
**전체 target-map cache를 굽지 않는 것이지 raycasting 자체를 생략하거나 계산 비용이 0인 것은 아닙니다.**
최종 pose는 공통 직접 raycast 경로로 density 전체를 재계산해 학습 내부 수치와 비교합니다.
legacy grid-cache evaluator와 별도 실행 경로이므로 물리 변경 시 parity 검증이 계속 필요합니다.

### 국소 보상과 PPO

현재 목표는 union coverage지만 학습 보상에는 v3의 shaping이 남아 있습니다. 최종 선택은 보상합이 아닌 coverage입니다.
covered target당 score 0.01로 `S_t=0.01*sum(c_j)`이며 team reward는 다음과 같습니다.

```text
r_team = 25*(S_t-S_(t-1)) + 0.1*S_t - 0.0*overlap_score
r_i = r_team + r_far_i + r_close_i + r_uncovered_i + r_edge_i + r_unique_i
r_buffer_i = clip(0.01*r_i, -10, 10)
```

- `r_far=min(nearest_other_sensor_distance/20,1)`; nearest distance가 5m 미만이면 추가 -5.
- `r_uncovered=1.5*clip(previous_nearest_uncovered_distance-current_distance,-20,20)`.
  XY 거리이며 미관측 집합 자체도 step마다 변합니다. 같은 고정 target을 추적하는 항이 아닙니다.
- 지도 경계 5m 이내에서 중심 반대 방향을 보면 `-2*max(-heading_dot_to_center,0)*(5-edge_distance)/5`.
- `r_unique=0.01*해당 센서만 보는 target 수`; overlap penalty coefficient는 0.

이 shaping은 coverage와 동일한 단일 최적화 목적을 이론적으로 보존한다고 검증한 보상이 아닙니다.
관련 ablation 및 단순 국소 탐색과의 비교가 필요합니다.

| PPO/평가 항목 | 실제 설정 |
|---|---|
| joint training steps / paired seeds | 2,000 / 0,1,2 |
| rollout / epochs / minibatch | 256 joint steps / 5 / 128 agent samples |
| learning rate / gamma / GAE lambda | 0.0003 / 0.999 / 0.95 |
| policy clip / value clip | 0.15 / 0.2 |
| initial entropy / update decay / minimum | 0.003 / 0.997 / 0 |
| masked random-action mixture | 0.005, sampling/update 모두 같은 분포 사용 |
| actor/critic optimizer / gradient clipping | 별도 Adam / 각각 norm 1.0 |
| deterministic 평가 | 100 step 간격, 매번 초기 배치로 복원 후 50 actions |
| 마지막 rollout | 남은 208 joint steps도 update; 합계 8 PPO updates |
| 한 run의 계산량 | training 6,000 agent transitions + 평가 20회 x 50 joint steps |

continuing task로 `done=False`이며 rollout 끝을 다음 critic 값으로 bootstrap합니다.
GAE는 `[time,agent]`에서 시간축을 따라 계산하고, 이후 minibatch로 펼칩니다.
수식은 `delta_ti=r_ti+gamma*V_(t+1,i)-V_ti`, `A_ti=delta_ti+gamma*lambda*A_(t+1,i)`입니다.
advantage는 joint rollout 전체 sample에서 정규화합니다. PPO ratio는 반드시 `[B]`이며
`exp(clip(logp_new-logp_old,-20,20))`를 clipped surrogate에 사용합니다. value clipping도 적용합니다.
평가 trajectory는 PPO buffer에 넣지 않고 학습 pose·reward/hint 상태를 복원합니다.
warm-spawn/plateau 관련 param 필드가 저장돼 있어도 이 local runner는 전체 spawn 탐색과 plateau schedule을 실행하지 않습니다.

### 저장값과 정책 성능의 차이

```text
deployment = argmax coverage among
    initial placement,
    stochastic training poses,
    each fixed-start deterministic evaluation endpoint
final_policy = final network의 fixed-start 50-step 평가 종료 배치
best_eval = 평가 이력의 최고 종료 배치
```

동일한 완전 pose의 동시 제안은 deployable best 후보에서 제외합니다.
초기 배치를 보존하므로 deployment coverage 비감소는 설계상 성질이지 정책 향상의 증거가 아닙니다.
local runner는 **yaw 후처리를 하지 않습니다**. 기존 v3의 후처리 포함 결과와 혼용하지 않습니다.
`policy.pt`는 final policy와 params/seed/initial pose/radius를 저장하지만 optimizer/rollout resume는 제공하지 않습니다.

로그 NPZ는 `step,coverage,best_coverage,reward,eval_step,eval_coverage`입니다.
JSON `refinement`에는 실제 params, 8개 update의 PPO 진단값, initial/final/best-eval coverage,
선택 출처, elapsed time, raycast 질의 수·시간, memoized XYZ 수, device가 있습니다.
상위 `{method}_coverage.npz`에는 전체 `target_points,density`, NPY에는 `[K,5]` pose가 들어갑니다.

### 논문에서 구분해야 하는 가설과 미검증 항목

현재 실험은 **폭 하나만 변하는 세 정형 topology에서 pose library 재사용이 가능한지**를 평가합니다.
보간이 단독 전이보다 일관되게 좋은지, MAPPO가 다른 국소 탐색보다 유리한지, 전체 재최적화보다
얼마나 빠른지, reference optimum에 얼마나 가까운지는 서로 별개의 가설입니다.

- 실제 OSM polyline/건물 footprint, 곡률·교차각·차로별 폭/길이 변형: TBD.
- 자동 prototype 검색, 센서 배분이 다른 library끼리의 보간, K별 library 완성 및 K 자동 선택: TBD.
- pretrained policy 전이, zero-shot, 전체 도시 조합, 차량/보행자 동적 occlusion: TBD.
- 이 과거 폭 실험의 full-candidate 인증은 미완료입니다. 최신 offset target 4개/K3,4는 full-grid gap 기준선을 확보했습니다.
  반복 wall-clock 예산별 random/local-search/MAPPO 비교는 TBD입니다.
- 반사·assignment metric·anchor·투영·reward shaping·반경·학습량·prototype 품질 ablation: TBD.
- 완전히 별도인 최종 holdout, 충분한 독립 geometry, 신뢰구간·검정·effect size: TBD.

15개 맵은 세 geometry family의 폭 변형이므로 15개 독립 실도로로 설명하지 않습니다.
paired seeds는 비교조건을 맞추는 설계이며 통계적 유의성이나 bitwise 동일 난수를 보장하지 않습니다.
prototype 최적성 인증이 없는 상태에서 목표 맵 결과를 near-optimal이라고 단정하지 않습니다.

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

## 오프라인 Prototype: 시나리오 0·1·2 파일럿

### 도로 상대 좌표 전이와 MAPPO 국소 보정

`prototype_refine.py`는 시나리오별로 다음 **6개 결과 슬롯**을 비교합니다.

| 초기 배치 | 보정 전 | 보정 후 |
|---|---|---|
| 7m prototype 단독 전이 | transfer_low | transfer_low_mappo |
| 14m prototype 단독 전이 | transfer_high | transfer_high_mappo |
| 두 prototype 보간 | interpolation | interpolation_mappo |

이 절은 2026-10-02의 개발 파일럿 기록입니다. 입력은 직선/T자/십자 라이브러리,
이 과거 파일럿의 목표는 각 topology의 폭 10m, K=3입니다. 이후 15맵 실험은 아래 확장 실험 절을 기준으로 봅니다.
절차는 **도로·설치선 대응 → 도로 상대 좌표 보간/전이 → 목표 설치선 투영 →
설치 제약 검사 → 직접 coverage 평가 → 국소 MAPPO 보정**입니다.
기존 결과와 원본 라이브러리는 덮어쓰지 않습니다.

- `prototype_transfer.py --matching road_relative`: 같은 진입로/인도 측면끼리 대응합니다.
  직선은 지도 중심 기준 종방향 거리, 교차로는 교차 영역 경계(`road_width/2`)에서 바깥으로
  향하는 거리를 사용합니다. 설치선으로부터의 작은 횡방향 offset, 높이, 상대 yaw, pitch를
  보간하고 새 맵의 기존 XYZ/yaw/pitch 격자로 투영합니다. yaw는 원형 보간입니다.
- 단독 전이도 **같은 도로 상대 좌표 변환**을 사용합니다. 교차로 경계 위치가 바뀌므로
  이전 `geometry` 모드의 단독 전이 수치와 달라질 수 있습니다.
- 기존 T자/십자 원본은 같은 설치선별 센서 배분이 맞지 않습니다. 이 경우 보간 및 그 MAPPO
  결과는 `unavailable`로 남습니다. 단독 전이로 대체하거나 0% 점수로 채우지 않습니다.
  따라서 원본 라이브러리 그대로는 18개 슬롯 중 14개 배치가 실행 가능하고 4개는 미생성입니다.
  아래의 보조 prototype 경로는 원본을 보존하고 설치선 배분을 맞춘 별도 라이브러리를 사용합니다.
- `src/methods/local_mappo.py`: 기존 v3의 actor/critic, PPO/GAE와 11개 행동·보상을 재사용합니다.
  각 초기 배치마다 정책/optimizer를 새로 만들고 시작 pose를 정확히 주입합니다.
  기본 파일럿은 **조건별 2,000 joint steps, seed=0, XY 반경 10m**, 2m XY 이동입니다.
  반경은 각 센서의 초기 위치 기준이며 높이/yaw/pitch는 기존 허용 범위를 사용합니다.
  인도 선이 단절되거나 2m 행동으로 도달할 수 없는 위치로 이동하는 기능은 없습니다.
- 새로운 맵의 전체 visibility cache를 생성/로드하지 않습니다. 공통 `MeshVisibility`에서
  요청받은 유효 XYZ의 LOS만 계산하고 **해당 학습 실행 내부에서만 메모이제이션**합니다.
  local-map 관측에 필요한 주변 위치도 계산량에 포함됩니다. 모든 위치의 visibility 최대값을
  미리 구하지 않도록 관측 정규화 분모는 전체 target 수로 고정합니다.
- 초기 배치, stochastic 탐색, 고정 초기값 deterministic 평가 중 최고 유효 배치를 저장합니다.
  초기 배치 보존에 따른 비감소는 학습 수렴의 증거가 아닙니다. 최종 정책 배치는 별도 NPY와
  `final_policy_coverage_pct`에 기록합니다. yaw 후처리, 기존 checkpoint 로드, plateau schedule은
  이 보정 경로에서 사용하지 않습니다. 정책 평가는 100 step마다 50 actions, 마지막 update 후에도 수행합니다.
- 단일 seed/폭 변형 파일럿입니다. 교차각·곡률 변형, zero-shot 정책 전이, near-optimal 보장은
  검증하지 않았습니다. 원본 MILP의 `best_known`을 인증된 최적해라고 해석하지 않습니다.

실행 전 계획 확인(학습 없음):

```bash
.venv/bin/python prototype_refine.py \
  --output results/prototype_relative_mappo_20261002 --steps 2000 --seed 0 --radius 10
```

실제 계산 명령(기존 output이면 실패하므로 반복 시 새 폴더 지정):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python prototype_refine.py \
  --output results/prototype_relative_mappo_20261002 \
  --steps 2000 --seed 0 --radius 10 --execute
```

`CUDA_VISIBLE_DEVICES=1`은 서버의 두 번째 GPU를 선택합니다. 다른 머신에서는 사용 가능한
GPU에 맞추거나 생략합니다. 설정 YAML이나 기존 official MAPPO CLI 기본값은 바꾸지 않습니다.

출력은 `manifest.json`, `comparison.csv`, `comparison.html`, `index.html`, 그리고
`s0/`, `s1/`, `s2/`의 결과 JSON/NPY/관측 density NPZ/회전 가능한 3D HTML입니다.
보정별 하위 폴더에는 `initial.npy`, `deployment.npy`, `final_policy.npy`, `best_eval.npy`,
`train_log.npz`, `policy.pt`가 저장됩니다. report의 refinement metadata에는 PPO 진단값,
실제 파라미터, 실행시간 및 직접 raycasting 수를 기록합니다. 정책 파일은 완전한 resume checkpoint가 아닙니다.
코드 snapshot/hash와 source prototype 사본도 저장합니다. 생성 결과는 ignored 경로입니다.

검증 명령:

```bash
.venv/bin/python tests/test_local_mappo.py
.venv/bin/python -m unittest discover -s tests -p 'test_prototype*.py' -v
.venv/bin/python tests/test_pipeline.py
```

국소 보정 테스트 6개, prototype 테스트 16개, 기존 파이프라인 테스트 7개 통과.
문법 검사와 3개 scenario 계획 확인 통과. 경량 테스트는 실제 작은 PPO update를 포함합니다.
실제 계산 완료 여부는 해당 실행의 `manifest.json`을 기준으로 확인합니다.

2026-10-02 실행 결과: **14개 배치 완료, 보간 대응 불가 4개**, manifest 상태는 `partial`입니다.
7개 MAPPO 보정 모두 2,000 joint training steps와 PPO update 8회가 수행됐습니다.
각 보정에는 별도로 deterministic 평가 20회 x 50 steps가 포함됩니다.

| 시나리오 | 7m 단독 → MAPPO 최고 배치 | 14m 단독 → MAPPO 최고 배치 | 보간 → MAPPO 최고 배치 |
|---|---:|---:|---:|
| 0 직선 | 99.778 → 99.778% | 99.815 → 99.815% | 99.917 → 99.917% |
| 1 T자 | 90.653 → 96.648% | 91.787 → 96.356% | 미생성: 설치선 배분 불일치 |
| 2 십자 | 80.843 → 81.836% | 80.073 → 84.002% | 미생성: 설치선 배분 불일치 |

직선의 세 대표 배치는 모두 초기값이 유지됐습니다. T자/십자의 개선값은 stochastic 탐색 중
최고 배치이며 최종 정책 평가값이 아닙니다. 최종 정책 coverage는 T자 92.915/94.322%,
십자 77.345/81.521%입니다. 단일 seed의 결과이므로 MAPPO의 수렴이나 일관된 개선을 주장하지 않습니다.

- [결과 및 회전 가능한 3D 목록](results/prototype_relative_mappo_20261002/index.html)
- [비교 그래프](results/prototype_relative_mappo_20261002/comparison.html)
- [18개 상태/수치 CSV](results/prototype_relative_mappo_20261002/comparison.csv)
- [14개 배치 재검산 기록](results/prototype_relative_mappo_20261002/validation.json)

재검산은 학습이나 전체 cache 생성 없이 수행합니다. 최종 policy coverage도 별도로 확인합니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python tests/validate_prototype_refinement.py results/prototype_relative_mappo_20261002
```

CSV/비교 그래프/3D 목록만 다시 생성하려면 다음 명령을 사용합니다. 학습과 원본 pose를
변경하지 않으며 자료 생성 코드 hash는 `report_source_hashes.json`에 별도로 기록합니다.

```bash
.venv/bin/python prototype_refine.py \
  --output results/prototype_relative_mappo_20261002 --report-only
```

### 설치선 배분을 맞춘 보조 Prototype

`prototype_compatible.py`는 T자/십자의 7m 배치를 기준으로 14m 보조 배치를 만듭니다.
각 `(진입로, 인도 측면)`의 센서 수를 동일하게 고정하고, 해당 설치선 위 위치·높이·yaw·pitch를
기존 전체 이산 격자에서 선택합니다. 현재 파일럿은 K=3, 그룹당 정확히 1대만 지원합니다.
이 제약은 보조 라이브러리 생성에만 적용되며 기존 official solver의 기본 문제는 바꾸지 않습니다.

- T자: east/left, south/left, west/right에 각 1대.
- 십자: east/right, north/left, west/left에 각 1대.
- top-M pruning 없음. coverage가 같은 후보의 제거는 **동일 설치선 그룹 안에서만** 수행합니다.
  서로 다른 그룹의 동일 mask를 합치면 할당 제약의 가능해를 잃을 수 있기 때문입니다.
- 기존 14m의 물리 visibility cache만 환경 source hash 확인 후 재사용합니다.
  전역 dedup된 기존 후보 archive는 사용하지 않고 후보를 다시 만듭니다.
- 그룹 순서별 GREEDY warm start를 만든 뒤 Gurobi에 정확한 그룹별 수 제약과 함께 제공합니다.
  solver 제한은 300초, 1 thread, MIP gap 0, soft memory 90GB입니다.
  후보 생성·행렬 구성은 300초에 포함되지 않으며 종료 처리로 solver runtime도 소폭 넘을 수 있습니다.

생성된 14m 보조 prototype의 직접/캐시/후보 coverage 검산은 일치했습니다.

| 보조 prototype | 후보 수 | 14m coverage | solver 상태 | 인증 |
|---|---:|---:|---|---|
| T자 | 59,603 | 92.321% | 9, time limit | constrained_best_known |
| 십자 | 52,964 | 85.759% | 9, time limit | constrained_best_known |

두 solve 모두 quota-GREEDY 초기 해를 개선하지 못했고 유한한 best bound를 얻지 못했습니다.
따라서 **제약부 feasible 배치이며 MILP 인증 최적해가 아닙니다.** 아래의 10m 평가와 위 14m
coverage는 서로 다른 맵의 수치입니다. 비유한 bound/gap은 JSON에서 `null`로 저장합니다.

재현 명령(실제 계산, 기존 output 경로에서는 실패):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_compatible.py \
  --reference results/prototype_junctions_20261002/s1_w7 \
  --template results/prototype_junctions_20261002/s1_w14 \
  --output results/prototype_compatible_20261002/s1_w14 --time-limit 300 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_compatible.py \
  --reference results/prototype_junctions_20261002/s2_w7 \
  --template results/prototype_junctions_20261002/s2_w14 \
  --output results/prototype_compatible_20261002/s2_w14 --time-limit 300 --execute

# --execute 없이 먼저 3개 시나리오의 보간 가능 여부를 확인
.venv/bin/python prototype_refine.py \
  --auxiliary-root results/prototype_compatible_20261002 \
  --output results/prototype_relative_mappo_compatible_20261002 \
  --steps 2000 --seed 0 --radius 10

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python prototype_refine.py \
  --auxiliary-root results/prototype_compatible_20261002 \
  --output results/prototype_relative_mappo_compatible_20261002 \
  --steps 2000 --seed 0 --radius 10 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_prototype_refinement.py results/prototype_relative_mappo_compatible_20261002
```

새 비교에서 직선의 두 원본과 모든 7m 원본은 그대로 사용하고 T자/십자의 14m 입력만
보조 prototype으로 바꿉니다. 모든 MAPPO는 이전 checkpoint 없이 다시 시작합니다.
보정 중에는 유효 설치 영역과 초기 XY 반경을 제한하며, 별도 그룹별 센서 수 제약을 강제하지 않습니다.

**비교 해석:** `transfer_high`는 온라인에서 14m prototype 하나만 사용하지만,
그 prototype 자체는 오프라인에서 7m 배분을 참고해 만들었습니다. 두 원본을 서로 독립적으로
최적화한 비교와 같지 않습니다. 기존 제약 없는 14m 전이 결과는 이전 폴더에 보존합니다.
보간을 가능하게 만들었다는 사실이 보간의 성능 우위나 near-optimality를 뜻하지 않습니다.

#### 보조 라이브러리 비교 결과: 18/18 완료

2026-10-03 00:02~00:07 KST에 실행했습니다. 폴더의 `20261002`는 작업 시작일 tag이며,
실제 시간은 `manifest.json`의 UTC timestamp로 기록했습니다. 3개 시나리오 x 6개 배치,
9개 MAPPO 보정(각 2,000 joint steps, seed 0, PPO update 8회)이 완료됐습니다.
아래는 **10m 새 맵, K=3**에서의 coverage입니다. 화살표 뒤는 초기값을 포함한 최고 배치입니다.

| 시나리오 | 7m 단독 → MAPPO 최고 | 14m 단독 → MAPPO 최고 | 보간 → MAPPO 최고 |
|---|---:|---:|---:|
| 0 직선 | 99.778 → 99.778% | 99.815 → 99.815% | 99.917 → 99.917% |
| 1 T자 | 90.653 → 96.648% | 91.717 → 94.056% | 91.698 → 96.375% |
| 2 십자 | 80.843 → 81.836% | 84.666 → 84.666% | 82.568 → 83.731% |

T자/십자의 14m 열은 **할당 제약이 있는 보조 prototype**입니다. 직선은 기존 원본입니다.
직선에서는 보간 초기값이 가장 높았지만 MAPPO가 이를 개선하지 못했습니다.
T자에서는 7m 단독+MAPPO, 십자에서는 14m 단독 초기값이 가장 높았습니다.
따라서 이번 결과로 보간이나 MAPPO의 일관된 우위를 주장할 수 없습니다.

최종 deterministic 정책의 coverage는 최고 배치와 다르며 다음과 같습니다.

| 시나리오 | 7m 초기화 정책 | 14m 초기화 정책 | 보간 초기화 정책 |
|---|---:|---:|---:|
| 0 직선 | 88.343% | 98.491% | 99.917% |
| 1 T자 | 92.915% | 88.916% | 93.473% |
| 2 십자 | 77.345% | 80.237% | 80.257% |

이는 새 정책을 짧게 학습한 단일 seed 실험입니다. 위 최고 배치는 best-seen search 결과이며
정책 수렴 또는 학습된 전이 정책 성능과 같지 않습니다. 새 10m 맵의 MILP 기준값은 아직
이 비교에 없으므로 near-optimal 여부는 **TBD**입니다.

- [10월 2일 보조 library의 18개 결과 및 회전 가능한 3D 목록](results/prototype_relative_mappo_compatible_20261002/index.html)
- [6개 방법 x 3개 시나리오 비교 그래프](results/prototype_relative_mappo_compatible_20261002/comparison.html)
- [coverage·중복·미관측·시간 CSV](results/prototype_relative_mappo_compatible_20261002/comparison.csv)
- [18개 직접 재검산 기록](results/prototype_relative_mappo_compatible_20261002/validation.json)

검산은 복사된 prototype으로 초기 전이/보간 배치를 재생성해 NPY와 비교하고, 보조 배분,
최종 설치 제약, 직접 raycast density, 국소 반경, PPO finite 로그와 최종 정책 coverage를 검사했습니다.
prototype 단위 테스트 20개(그룹 MILP 전수조사 비교 포함), local MAPPO 6개,
기존 pipeline 7개 및 문법 검사 통과. 생성 파일은 ignored 경로이며 clone에는 포함되지 않습니다.

### 미사용 도로 폭 15개 확장 실험

`prototype_benchmark.py`는 기존 7m/14m 라이브러리를 고정하고 폭 변화에 대한 성능을 평가합니다.
현재 실행 폴더는 `results/prototype_width_sweep_20261003/`입니다.
2026-10-03 11:26~12:11(KST)에 **15/15개 맵을 완료**했으며 실패한 맵은 없습니다.
전체 병렬 작업 시간은 약 45.08분입니다. 이는 개별 방법의 계산시간이나 speedup 수치가 아닙니다.
실행 당시 코드와 설정은 해당 폴더의 `source/`와 `manifest.json`으로 확인합니다.

| 항목 | 고정 조건 |
|---|---|
| topology | 직선(s0), T자(s1), 십자(s2) |
| 새 폭 | 8, 9, 11, 12, 13m: 총 15개 맵 |
| 기존 개발용 폭 | 10m: 이번 집계에서 제외 |
| 원본 | 기존 7m/14m; T자/십자의 14m는 설치선 배분을 맞춘 보조 prototype |
| K / LiDAR | 3 / 기존 VELODYNE64_R30 |
| 보정 전 | 7m 단독, 14m 단독, 두 prototype 보간 |
| MAPPO | 세 초기화 각각 2,000 joint steps, seed 0/1/2, XY 반경 10m |
| 기준선 | 새 맵의 GREEDY 및 Gurobi MILP, 위치별 top-10 후보 |
| MILP | 300초, relative gap 0.05, 1 thread, worker별 soft memory 26GB |
| 병렬 실행 | GPU 1/2/3에서 맵별 worker 3개; K/후보/보상 기본 YAML은 변경하지 않음 |

MAPPO 보정은 **15 x 3초기화 x 3seeds = 135회**입니다. 각 seed에서 학습 전 세 배치도 저장하지만,
동일한 deterministic 초기 배치를 반복 trial처럼 평균내지 않습니다. 최종 `trials.csv`는
보정 전 45개 + MAPPO 135개 + GREEDY/MILP 30개 = **210행**,
`summary.csv`는 15개 맵 x 8개 방법 = **120행**이 완전성 기준입니다.
MAPPO 표준편차는 표본 SD(`ddof=1`)이며 최고 배치와 최종 정책 성능을 분리합니다.

계획 확인과 실행:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_benchmark.py \
  --output results/prototype_width_sweep_20261003

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_benchmark.py \
  --output results/prototype_width_sweep_20261003 \
  --widths 8 9 11 12 13 --scenarios 0 1 2 --seeds 0 1 2 \
  --steps 2000 --radius 10 --baseline-topm 10 --time-limit 300 --gap 0.05 \
  --gpus 1 2 3 --execute

# 기존 결과만 집계/그래프 재생성, 학습/solve 없음
.venv/bin/python prototype_benchmark.py \
  --output results/prototype_width_sweep_20261003 --report-only

# 완료 후 전체 배치/정책/기준선/집계표 독립 재검산
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tests/validate_width_benchmark.py results/prototype_width_sweep_20261003

tail -f results/prototype_width_sweep_20261003/s1_w8.log
```

새 실행에는 새 output root가 필요합니다. 자동 resume는 구현하지 않았습니다.
`--baseline-topm 0`은 전체 orientation 기준선 모드지만, 메모리 안전을 위해 GPU/worker 한 개만
허용합니다. 이번 실행은 그 모드가 아니며, top-10 MILP는 전이/MAPPO 전체 공간의 상한이 아닙니다.

출력 구조:

```text
prototype_width_sweep_20261003/
  manifest.json / hardware.json / source/
  trials.csv / summary.csv / paired_differences.csv / validation.json / final_validation.json
  index.html / coverage_s0.html / coverage_s1.html / coverage_s2.html
  s0_w8.log ... s2_w13.log
  s0_w8/ ... s2_w13/
    manifest.json
    seed0/ seed1/ seed2/    # 각각 기존 6배치 결과, 정책, 로그, 3D, 검산 기록
    baselines/             # config, 새 visibility cache, GREEDY/MILP 결과, solver log, 3D
```

기준선은 새 맵의 전체 visibility cache와 후보를 만들지만, MAPPO에는 그 cache를 전달하지 않습니다.
MAPPO는 실행별 on-demand LOS만 사용합니다. 각 seed 결과를 직접 raycasting으로 재검산하고,
원본 prototype hash, 보간 재생성, 설치 제약, 국소 반경, PPO 로그를 확인합니다.
실패/미완료 맵은 0으로 채우지 않고 집계에서 제외하며 root 상태는 `partial` 또는 `failed`입니다.

시간은 후보/cache 구성, solver 전체 호출(행렬 구성 포함), 직접 평가, MAPPO 초기화/학습/평가를
나눠 기록합니다. `compute_component_sum_sec`는 **계산 요소 합계**이며 모든 prototype 파일 로드와
HTML export를 포함한 독립 wall-clock latency는 아닙니다. worker 전체 시간은 `manifest.json`에 있습니다.
병렬 실행과 GPU 기종 차이가 있으므로 이 수치만으로 공정한 speedup을 주장하지 않습니다.

`paired_differences.csv`는 보간과 두 단독 전이 중 최고값의 차이를 seed별로 기록합니다.
여기서 최고 단독값은 결과를 본 뒤 고른 분석 기준이며, 새 맵에서 평가 비용 없이 선택할 수 있는
실제 배치 알고리즘으로 간주하지 않습니다. 3 seeds는 예비 반복이며 통계적 우위 증명이 아닙니다.
기하 변화는 폭만 바꾸고 차로 수는 유지합니다. OSM 실도로 형상이나 곡률/교차각 변형 실험은 아닙니다.
기존 0.5m 설치 격자와 2m bin의 첫 위치 선택을 그대로 쓰므로 폭에 따라 후보 수가 불연속적으로
변합니다. 실제 직선 XYZ base는 11m에서 800개, 8/9/12/13m에서 400개입니다.
따라서 성능 차이를 연속적인 도로 폭 변화만의 효과로 해석하지 않으며 `candidate_stats.json`도 함께 봅니다.
이 결과를 보고 보간 규칙이나 하이퍼파라미터를 변경하면, 별도의 최종 미사용 검증 맵이 필요합니다.

#### 확장 실험 결과와 해석

- [전체 결과와 회전 가능한 3D 목록](results/prototype_width_sweep_20261003/index.html)
- [맵별 평균/표본 SD](results/prototype_width_sweep_20261003/summary.csv), [trial 원자료](results/prototype_width_sweep_20261003/trials.csv)
- [전체 독립 재검산 기록](results/prototype_width_sweep_20261003/final_validation.json)

아래는 각 topology의 다섯 폭을 동일 가중치로 평균한 coverage(%)입니다.
MAPPO 열은 각 맵의 3 seeds 평균을 다시 평균한 **best-seen deployment**이며 최종 정책 성능이 아닙니다.
맵별 수치와 SD는 위 CSV에 있습니다. 생성 결과는 ignored 경로라 저장소 clone에는 포함되지 않습니다.

| topology | 7m 전이+MAPPO | 14m 전이+MAPPO | 보간+MAPPO | GREEDY top-10 | MILP top-10 |
|---|---:|---:|---:|---:|---:|
| 직선 | 99.719 | 99.852 | 99.868 | 99.893 | 99.893 |
| T자 | 96.338 | 96.136 | 96.205 | 92.235 | 96.588 |
| 십자 | 82.747 | 85.114 | 84.334 | 80.515 | 82.545 |

두 단독 전이 중 높은 coverage와 비교하면, **보정 전 보간은 1개 우세 / 6개 동률 / 8개 열세**입니다.
MAPPO 보정 후 방법별 3 seeds 평균끼리 비교하면 **3개 우세 / 3개 동률 / 9개 열세**입니다.
이는 사후 최고 단독값과의 비교이며, seed별 최고값을 먼저 고르는 `paired_differences.csv`와 집계 순서가 다릅니다.
보간 초기화에서 MAPPO 보정으로 얻은 평균 개선은 직선 0.000%p, T자 4.742%p, 십자 0.893%p입니다.
직선은 초기부터 coverage가 포화에 가까우므로 보정 효과가 작습니다.

MILP는 15개 중 status 2가 10개, 시간 제한 status 9가 5개입니다.
저장 gap이 0인 것은 3개뿐이며, 유한한 gap을 확보하지 못한 4개는 JSON에서 null로 기록했습니다.
status 2만으로 exact optimum이라 부르지 않으며, top-10 후보 기준선과 전이/MAPPO의 탐색 공간 차이도 유지됩니다.
따라서 십자의 보간+MAPPO 값이 이 MILP 기준선보다 높다고 전역 최적해를 능가했다고 해석하지 않습니다.

**결론:** 설치 제약을 만족하는 전이/보간과 국소 MAPPO 보정 파이프라인은 작동하지만,
보간이 단독 전이보다 일관되게 좋은 초기화를 준다는 가설은 이번 결과로 지지되지 않습니다.
MAPPO 보정 후에도 단독 전이가 더 좋은 맵이 많습니다. 다음 단계는 실패 맵의 센서 대응과
설치선 배분을 분석하고, 동일 계산 예산의 국소 탐색과 비교하는 것입니다.

검산은 135개 MAPPO run의 배치/정책/로그, 30개 기준선의 직접 raycast density,
고정 prototype hash, 설치 제약, 210행 원자료 및 120행 평균/표본 SD 일치를 통과했습니다.
이는 저장 결과와 계산 경로의 일치 검증이며 학습 수렴이나 일반화 성능의 증명은 아닙니다.

#### 전체 15개 맵의 보정 전 결과

단위는 coverage %입니다. 7m/14m는 목표 폭이 아니라 사용한 원본 prototype을 뜻합니다.
동일한 초기 배치를 seed별 반복 표본으로 세지 않습니다.

| scenario | 새 폭 m | 7m 단독 전이 | 14m 단독 전이 | 보간 |
|---|---:|---:|---:|---:|
| 0 직선 | 8 | 99.893 | 99.762 | 99.893 |
| 0 직선 | 9 | 99.875 | 99.792 | 99.906 |
| 0 직선 | 11 | 99.633 | 99.833 | 99.833 |
| 0 직선 | 12 | 99.439 | 99.848 | 99.848 |
| 0 직선 | 13 | 99.049 | 99.861 | 99.861 |
| 1 T자 | 8 | 90.469 | 91.545 | 90.235 |
| 1 T자 | 9 | 91.120 | 91.795 | 91.780 |
| 1 T자 | 11 | 90.453 | 91.558 | 91.506 |
| 1 T자 | 12 | 89.897 | 92.596 | 91.338 |
| 1 T자 | 13 | 90.360 | 92.454 | 92.454 |
| 2 십자 | 8 | 79.364 | 84.615 | 80.141 |
| 2 십자 | 9 | 79.316 | 84.580 | 81.967 |
| 2 십자 | 11 | 80.729 | 84.293 | 84.236 |
| 2 십자 | 12 | 80.557 | 85.841 | 85.040 |
| 2 십자 | 13 | 80.354 | 85.826 | 85.826 |

#### MAPPO 보정 결과와 새 맵 기준선

단위 %, MAPPO는 **최고 배치의 3 seeds 평균 ± 표본 SD**입니다. SD=0은 보존된 초기값에서
최고 배치가 개선되지 않은 경우에도 발생하므로 정책의 안정적 수렴을 뜻하지 않습니다.
GREEDY/MILP는 반복 학습이 없는 단일 실행값입니다. 표의 반올림 값으로 우열을 다시 판정하지 않습니다.

| scenario | 폭 m | 7m + MAPPO | 14m + MAPPO | 보간 + MAPPO | GREEDY | MILP |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 99.893 ± 0.000 | 99.849 ± 0.076 | 99.893 ± 0.000 | 99.893 | 99.893 |
| 0 | 9 | 99.896 ± 0.018 | 99.830 ± 0.066 | 99.906 ± 0.000 | 99.906 | 99.906 |
| 0 | 11 | 99.781 ± 0.146 | 99.864 ± 0.053 | 99.833 ± 0.000 | 99.925 | 99.925 |
| 0 | 12 | 99.644 ± 0.217 | 99.856 ± 0.013 | 99.848 ± 0.000 | 99.879 | 99.879 |
| 0 | 13 | 99.382 ± 0.391 | 99.861 ± 0.000 | 99.861 ± 0.000 | 99.861 | 99.861 |
| 1 | 8 | 94.566 ± 1.351 | 95.472 ± 1.724 | 95.243 ± 1.419 | 92.136 | 92.136 |
| 1 | 9 | 96.370 ± 0.419 | 95.849 ± 1.233 | 96.320 ± 0.991 | 92.114 | 99.808 |
| 1 | 11 | 96.539 ± 0.566 | 96.625 ± 0.475 | 96.646 ± 0.819 | 91.535 | 91.535 |
| 1 | 12 | 96.634 ± 0.618 | 96.841 ± 0.405 | 96.564 ± 0.810 | 92.753 | 99.749 |
| 1 | 13 | 97.579 ± 0.412 | 95.892 ± 1.572 | 96.252 ± 0.929 | 92.637 | 99.712 |
| 2 | 8 | 80.506 ± 1.979 | 84.615 ± 0.000 | 82.452 ± 1.800 | 79.345 | 84.757 |
| 2 | 9 | 82.679 ± 0.845 | 84.598 ± 0.031 | 83.397 ± 0.742 | 80.921 | 85.659 |
| 2 | 11 | 82.809 ± 1.758 | 84.688 ± 0.525 | 84.928 ± 0.428 | 80.602 | 80.602 |
| 2 | 12 | 83.766 ± 0.252 | 85.841 ± 0.000 | 85.071 ± 0.053 | 80.533 | 80.533 |
| 2 | 13 | 83.977 ± 0.617 | 85.826 ± 0.000 | 85.826 ± 0.000 | 81.173 | 81.173 |

#### 최종 정책 평가와 탐색 최고값의 차이

아래는 5개 폭 x 3 seeds의 **최종 정책 fixed-start 50-step endpoint** coverage 평균입니다.
원자료는 `summary.csv`의 `final_policy_mean`이며 위 deployment 표와 다른 측정값입니다.

| topology | 7m 초기화 정책 | 14m 초기화 정책 | 보간 초기화 정책 |
|---|---:|---:|---:|
| 직선 | 92.218 | 95.635 | 94.278 |
| T자 | 93.113 | 92.319 | 91.506 |
| 십자 | 79.359 | 79.744 | 81.090 |

최종 정책이 탐색 최고 배치를 재현하지 못하는 차이가 큽니다. 따라서 현재 결과는
**RL을 사용한 배치 탐색의 성능**이지, 학습된 정책의 안정적 배치 생성이나 수렴을 입증하지 않습니다.
일부 최종 정책은 초기 배치보다 낮습니다. 최고 배치 보존이 이 사실을 숨기지 않도록 두 표를 함께 보고합니다.

#### Target·후보 수와 solver 인증

아래 bound는 target 개수 단위이며 gap은 **비율**입니다(0.05=5%). N은 총 target 수,
M은 top-10 및 exact-mask dedup 후 후보 수입니다. null은 유한한 인증값을 얻지 못했다는 뜻입니다.
status 2는 설정 tolerance 내 종료, 9는 time limit입니다.

| scenario | 폭 m | N | M | status | incumbent | best bound | relative gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 8400 | 3994 | 2 | 8391 | 8400 | 0.001073 |
| 0 | 9 | 9600 | 3992 | 2 | 9591 | 9600 | 0.000938 |
| 0 | 11 | 12000 | 7973 | 2 | 11991 | 12000 | 0.000751 |
| 0 | 12 | 13200 | 3988 | 2 | 13184 | 13200 | 0.001214 |
| 0 | 13 | 14400 | 3984 | 2 | 14380 | 14400 | 0.001391 |
| 1 | 8 | 12360 | 5588 | 9 | 11388 | 12360 | 0.085353 |
| 1 | 9 | 14076 | 5585 | 2 | 14049 | 14049 | 0 |
| 1 | 11 | 17472 | 10993 | 9 | 15993 | null | null |
| 1 | 12 | 19152 | 5419 | 2 | 19104 | 19104 | 0 |
| 1 | 13 | 20820 | 5413 | 2 | 20760 | 20760 | 0 |
| 2 | 8 | 16224 | 7187 | 2 | 13746 | 13914 | 0.012222 |
| 2 | 9 | 18444 | 7184 | 2 | 15782 | 15905 | 0.007794 |
| 2 | 11 | 22812 | 14027 | 9 | 18387 | null | null |
| 2 | 12 | 24960 | 6855 | 9 | 20101 | null | null |
| 2 | 13 | 27084 | 6847 | 9 | 21985 | null | null |

원자료는 `s{scenario}_w{width}/baselines/milp_top10.json`의 `solver` 및 `candidate_stats.json`입니다.
새 기준선은 GREEDY warm start보다 낮은 해이면 GREEDY를 채택하고 `selection_source`에 남깁니다.
GREEDY와 같은 수치라고 해서 solver가 아무 연산도 안 했거나 최적성이 증명됐다고 추정하지 않습니다.

#### 통계와 표 재현

seed별 coverage를 `C_s`라고 하면 `mean=sum(C_s)/3`, `SD=sqrt(sum((C_s-mean)^2)/2)`입니다.
모든 폭의 평균은 target 수나 교통량으로 가중하지 않은 기술통계입니다.
`max(mean(low),mean(high))`와 `mean(max(low_s,high_s))`는 일반적으로 다릅니다.
현재 3/3/9 승패는 전자, `paired_differences.csv`는 후자를 만들기 위한 seed별 차이입니다.
deterministic 초기 비교 차이가 seed별 CSV에 반복돼 있어도 독립 관측치가 늘어난 것은 아닙니다.
신뢰구간이나 통계 검정을 시행한 결과는 아직 없습니다.

표 확인용 명령(학습/solve/파일 변경 없음):

```bash
.venv/bin/python - <<'PY'
import csv
from collections import defaultdict
from statistics import mean
from pathlib import Path

root = Path('results/prototype_width_sweep_20261003')
rows = list(csv.DictReader((root / 'summary.csv').open()))
groups = defaultdict(list)
maps = defaultdict(dict)
for row in rows:
    groups[(int(row['scenario']), row['method'])].append(row)
    maps[(row['scenario'], row['width'])][row['method']] = float(row['coverage_mean'])
    print(row['scenario'], row['width'], row['method'], row['coverage_mean'], row['coverage_sd'])
for key, values in sorted(groups.items()):
    policy = [float(v['final_policy_mean']) for v in values if v['final_policy_mean']]
    print(key, 'deployment:', mean(float(v['coverage_mean']) for v in values),
          'final policy:', mean(policy) if policy else None)
for suffix in ['', '_mappo']:
    diffs = [v['interpolation' + suffix] - max(v['transfer_low' + suffix], v['transfer_high' + suffix])
             for v in maps.values()]
    print(suffix, 'win/tie/loss:', sum(x > 1e-7 for x in diffs),
          sum(abs(x) <= 1e-7 for x in diffs), sum(x < -1e-7 for x in diffs))
PY
```

#### 재현 환경과 검증 범위

실행 환경은 해당 root의 `hardware.json`이 기준입니다. Python 3.12.3, NumPy 2.4.3,
SciPy 1.17.1, PyTorch 2.10.0, Open3D 0.19.0, Gurobi 13.0.1, Plotly 6.6.0, PyYAML 6.0.3입니다.
GPU 1/2는 RTX 3090, GPU 3은 RTX 3090 Ti, driver는 580.173.02이며 RAM은 약 124.5GiB입니다.
이 hardware snapshot의 CPU 항목은 x86_64만 기록하므로 구체 CPU 모델은 이 파일만으로 재현할 수 없습니다.
GPU별 성능 차이와 세 worker의 CPU/RAM 경합 때문에 현재 시간 합계로 방법 간 speedup을 주장하지 않습니다.
원본 library 구축 비용도 온라인 국소 보정 시간에 포함되지 않습니다.

```bash
.venv/bin/python -m py_compile prototype_offline.py prototype_compatible.py \
  prototype_transfer.py prototype_refine.py prototype_benchmark.py \
  src/methods/local_mappo.py tests/validate_width_benchmark.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s tests -p 'test_prototype*.py' -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_local_mappo.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_pipeline.py
```

2026-10-03 완료 시 prototype 25개, local MAPPO 6개, 기존 pipeline 7개 테스트 및 문법 검사 통과 기록이 있습니다.
단위 테스트는 toy solve/PPO를 포함하며 큰 실험 재실행은 아닙니다.
`final_validation.json`은 실제 전체 결과 검산 기록, `validation.json`은 runner 집계 완전성 기록입니다.
`browser_checks/final.json`은 세 비교 그래프의 8 traces x 5 widths, desktop/mobile, 81개 index 링크를 확인한 기록입니다.
`browser_checks/scenes.json`은 대표 3D 세 맵의 desktop/mobile canvas 비어 있지 않음·카메라 회전·오류 검사를 기록합니다.
이 UI 검사는 전체 3D 파일 하나하나의 시각 검수가 아닙니다.
실행 종료 후 runner에 내부 worker의 명시적 `--execute` guard와 진행 메시지를 추가했으므로,
학습 당시 코드는 current checkout이 아닌 결과의 `source/` snapshot/hash로 식별합니다.
2026-10-06 문서 갱신에서는 README의 120개 coverage/SD 셀과 15개 solver 행을 원본 CSV/JSON에
대조했고 최신 결과 링크 7개, code fence 짝, `git diff --check`를 확인했습니다.
공통/최신 prototype 코드의 `py_compile`과 `tests/test_pipeline.py` 7개도 다시 통과했습니다.
이는 문서·경량 회귀 확인이며 135개 학습을 다시 실행하거나 최신 결과물을 덮어쓴 것이 아닙니다.

### 새 설치 영역 기준 보간: geometry 모드

2026-10-02에 `prototype_transfer.py --matching geometry`를 추가했습니다.
설치 영역은 **새 맵의 공통 환경에서 생성**하며, 이전 설치선의 가지/좌우 이름이 같아야 한다는
센서 대응 제한을 해제합니다. 기존 인도 중심선 부근 설치 판정, XYZ/yaw/pitch 격자,
센서 스펙과 coverage 계산은 변경하지 않았습니다. 기존 재현을 위해 기본 모드는 `strict`입니다.

절차:

1. 같은 topology, K, 실제 센서 수 및 폭 이외의 환경 조건을 확인합니다.
2. 도로 폭에 따른 설치선 offset을 반영한 공통 좌표계에서 센서끼리 Hungarian matching합니다.
   비용은 `(dx/100)^2+(dy/100)^2+(dz/6)^2+(circular_dyaw/180)^2+(dpitch/90)^2`입니다.
   T자는 좌우 반사, 십자는 좌우/상하 반사를 검토합니다. 90도 회전은 추가하지 않았습니다.
3. 두 배치를 새 폭으로 옮긴 뒤 `alpha=(10-7)/(14-7)`로 위치·pitch와 원형 yaw를 보간합니다.
4. 새 맵의 **전체 유효 XYZ 후보**로 투영합니다. 각 XYZ에서 가장 가까운 허용 yaw/pitch를
   사용하고, 완전히 같은 pose가 중복되지 않도록 전체 XYZ 이동거리 제곱합을 최소화하는
   할당을 구합니다. 같은 pole에 서로 다른 pose를 금지하는 제약을 새로 넣지는 않았습니다.
5. 최종 pose만 직접 raycasting합니다. 대응·투영 과정에는 coverage를 사용하지 않습니다.

이 투영은 센서 coverage 최적화가 아니라 기하학적인 유효성 보정입니다. 차도 위의 raw 보간점이
생길 수 있지만 저장·평가하는 최종 배치는 설치 가능한 후보에 있어야 합니다. `report.json`에
raw pose, 대응된 source pose, target 가지/설치선, XYZ 투영 거리 및 yaw 변화량을 기록합니다.
원본 라이브러리와 과거 strict 결과는 덮어쓰지 않았습니다.

| 10m 맵, K=3 | 보간 + 유효 영역 투영 | 7m 단독 전이 | 14m 단독 전이 |
|---|---:|---:|---:|
| T자 | 75.982256% | 89.746515% | 92.845374% |
| 십자 | 72.558140% | 79.127907% | 81.540698% |

**보간은 생성됐지만 두 사례 모두 단독 전이보다 낮았습니다.** T자는 최대 약 5.04m/66.43도,
십자는 약 4.60m/35도 투영 변화가 생겼습니다. 동일한 기하 비용 가중치를 두 사례에 사용했고
coverage 결과를 본 후 대응 가중치를 조정하지 않았습니다. `complete`는 산출물 생성 완료이며
성능 개선·최적성·RL 수렴을 뜻하지 않습니다. 소스 prototype도 최적성 미인증입니다.

- [T자 보간 3D](results/transfer_t_w10_k3_geometry_20261002/interpolation.html) / [report](results/transfer_t_w10_k3_geometry_20261002/report.json)
- [십자 보간 3D](results/transfer_cross_w10_k3_geometry_20261002/interpolation.html) / [report](results/transfer_cross_w10_k3_geometry_20261002/report.json)

각 폴더의 `interpolation.npy`, `transfer_low.npy`, `transfer_high.npy`는 분리된 초기 배치입니다.
보간 후 MAPPO 국소 보정 연결·학습은 **To be added**이며 이번에는 실행하지 않았습니다.
기존 MAPPO의 행동 공간이나 초기화 API를 변경한 것도 아닙니다. 새 맵의 전체 visibility cache와
전체 orientation별 mask는 만들지 않고, 투영용 XYZ 목록과 세 배치만 사용합니다.

재현 명령(먼저 `--execute` 없이 확인, 실제 재실행에는 새 output 경로 사용):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_transfer.py \
  --low results/prototype_junctions_20261002/s1_w7 \
  --high results/prototype_junctions_20261002/s1_w14 \
  --width 10 --k 3 --matching geometry \
  --output results/transfer_t_w10_k3_geometry_20261002 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_transfer.py \
  --low results/prototype_junctions_20261002/s2_w7 \
  --high results/prototype_junctions_20261002/s2_w14 \
  --width 10 --k 3 --matching geometry \
  --output results/transfer_cross_w10_k3_geometry_20261002 --execute

.venv/bin/python tests/test_prototype_transfer.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tests/test_pipeline.py
```

생성 결과는 ignored `results/` 아래에 저장되므로 저장소 clone에 자동 포함되지 않습니다.
검증: 문법 검사, 보간 테스트 10개 및 기존 pipeline 7개 통과. 여섯 저장 배치의 density/coverage를
직접 재검산했고 기존 단독 전이 네 배치와 직선 strict 배치 세 개가 이전 NPY와 동일했습니다.
새 보간 HTML 두 개는 PC/모바일에서 canvas 렌더링·회전·가로 overflow·JS 오류를 확인했습니다.

### 시나리오 1·2 교차로 폭 변형 파일럿 (과거 strict 규칙)

`prototype_junctions.py`는 T자(s1), 십자(s2)에 대해 7m/14m 오프라인 라이브러리와
미사용 10m 맵 보간 평가를 순차 실행합니다. K=3, solver 300초/90GB, top-M 해제 조건입니다.
캐시 생성/후보/행렬 생성 시간은 solver 제한과 별도입니다. 새 맵에 전체 visibility 캐시를
만들지 않고 보간 및 단독 전이 배치만 직접 raycasting합니다. MAPPO 학습은 포함하지 않습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python prototype_junctions.py \
  --output results/prototype_junctions_20261002
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_junctions.py \
  --output results/prototype_junctions_20261002 --execute
```

첫 명령은 오프라인 4개 조합의 dry-run입니다. 두 번째는 실제 계산이며 기존 output을 거부합니다.
각 하위 실행의 명령과 종료코드는 manifest, stdout/stderr는 `s1_w7.log` 등의 파일에 남습니다.
실행 중 상태와 최종 완료 여부는 이 파일들을 확인합니다.
현재 결과 요약과 3D 링크는 [SUMMARY.md](results/prototype_junctions_20261002/SUMMARY.md)에 있습니다.
원본 실패 로그와 manifest를 덮어쓰지 않으며, T자 재평가 결과는 `transfer_s1_review/`로 분리했습니다.

2026-10-02 완료 수치(K=3, 실제 설치 3대):

| 시나리오 | 7m library coverage | 14m library coverage | 10m: 7m 단독 전이 | 10m: 14m 단독 전이 | 10m 보간 |
|---|---:|---:|---:|---:|---|
| 1 T자 | 90.613243% | 92.307350% | 89.746515% | 92.845374% | 현 대응 규칙에서 불가 |
| 2 십자 | 79.377682% | 80.931332% | 79.127907% | 81.540698% | 현 대응 규칙에서 불가 |

10m target 수는 T자 15,780개, 십자 20,640개입니다. library 열과 전이 열은 서로 다른 맵의
coverage이므로 직접 성능 향상으로 해석하지 않습니다. 네 offline 실행은 모두 status 9로 끝났고,
로그상 GREEDY 초기해를 개선하지 못했습니다. 유한한 best bound/gap이 없어 **best_known**이며
전역 최적해나 near-optimal prototype이라는 인증은 없습니다. 실제 solver 로그는 presolve 중
제한을 넘겨 종료한 경우를 포함합니다. 300초는 정확한 전체 wall-clock 상한이 아닙니다.

두 전이 report의 `partial`은 보간이 성립하지 않았다는 뜻입니다. 초기 T자 실행의 종료코드 1은
원본 manifest에 유지했고, 오류 처리 보완 후 `transfer_s1_review/`에 단독 전이 결과를 저장했습니다.
십자 `transfer_s2/`는 정상 종료했지만 보간은 `unavailable`입니다. 완전한 보간 실험 성공이 아닙니다.
각 원본 배치의 cached/direct raycast 관측 mask 일치를 확인했으며 새 맵에서는 배치당
3개 pose만 raycasting했습니다. 새 맵 MILP 기준값과 MAPPO 보정은 **To be added**입니다.

교차로 좌표는 중심 (50,50)에서 바깥 방향을 s, 왼쪽 법선을 d로 정의합니다.
T자는 east/west/south, 십자는 east/west/south/north 가지를 사용합니다.
기존 환경의 road normal 판정으로 가지와 설치선을 배정하고, 각 pose에 heading과 상대 yaw를 저장합니다.
T자는 좌우 반사만, 십자는 좌우/상하 반사를 허용하며 도로 가지·설치선이 맞는 센서끼리 대응합니다.
구성이 맞지 않으면 보간은 `unavailable`, report는 `partial`로 기록합니다.
이때 7m/14m 단독 전이는 별도로 평가·저장하며 다른 가지의 센서를 몰래 평균내지 않습니다.
보간 불가를 coverage 0%로 넣거나 단독 전이를 보간 성공으로 표시하지 않습니다.
geometry/canonical 변환은 prototype runner에서만 확장했고 공통 물리 환경은 변경하지 않았습니다.
기존 직선 경로도 유지합니다. 실제 OSM 교차로 복원이나 임의 각도 교차로 전이는 아닙니다.

T자와 십자 모두 7m/14m 해의 가지별 설치선 구성이 다릅니다. 허용된 반사로도 일대일 대응되지
않으므로 현 규칙에서는 보간할 수 없습니다. 도로 폭만 가깝다고 배치가 연속적으로 변한다고
가정할 수 없다는 파일럿의 한계입니다. 같은 배치 구성별 prototype을 여러 개 보관하거나
이산적인 가지 배정과 연속 pose 보정을 분리하는 후속 검토가 필요합니다.
이번 실행에는 그런 보정이나 MAPPO 학습을 추가하지 않았습니다.

T자 단독 전이 재평가와 요약 재생성 명령:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_transfer.py \
  --low results/prototype_junctions_20261002/s1_w7 \
  --high results/prototype_junctions_20261002/s1_w14 \
  --width 10 --k 3 --output results/prototype_junctions_20261002/transfer_s1_review --execute

.venv/bin/python prototype_junctions.py --output results/prototype_junctions_20261002 \
  --report-only --transfer-s1 transfer_s1_review

.venv/bin/python tests/test_prototype_offline.py
.venv/bin/python tests/test_prototype_transfer.py
.venv/bin/python tests/test_prototype_junctions.py
```

재평가는 기존 output을 거부하므로 다시 계산하려면 새 경로를 사용합니다.
`--report-only`는 기존 JSON에서 `SUMMARY.md`/`summary.json`만 재생성하며 solver를 실행하지 않습니다.
위 prototype 테스트는 각각 4/8/2개입니다. 기존 pipeline 7개도 통과했고,
확장 코드로 생성한 직선 세 배치는 이전 저장 NPY와 동일했습니다.
네 library pose의 설치 격자·canonical 왕복 변환을 재검사했고, 10m 단독 전이 네 배치의
저장 density/coverage를 직접 raycasting으로 재계산해 일치를 확인했습니다.
네 전이 HTML은 Playwright로 desktop/mobile 각각 렌더링, 비어 있지 않은 canvas,
마우스 회전, 가로 overflow 및 JavaScript 오류 여부를 확인했습니다.
3D는 정적 PNG가 아니라 각 `transfer_low.html`/`transfer_high.html`을 브라우저에서 엽니다.

### 직선 도로 폭 변형과 보간 파일럿 (K=3)

2026-10-02에 기존 7m 직선 도로와 합성 14m 직선 도로로 라이브러리를 구성하고,
라이브러리에 포함하지 않은 10m 직선 도로에 배치를 전이했습니다. 이번 범위는 **K=3만**입니다.
14m/10m 맵은 실제 OSM 장소를 복원한 것이 아니라 시나리오 0의 폭만 바꾼 합성 맵입니다.
지도 크기, 차로 수(2개), 인도, 건물 높이, target 높이, 센서 및 설치 격자는 고정했습니다.
차로 수를 도로 폭에서 다시 추정하지 않았으므로 교통학적으로 현실적인 차로 설계 실험은 아닙니다.
공통 `MapGeometry`를 사용하며 기존 시나리오 JSON이나 연구 설정을 덮어쓰지 않았습니다.

| 구분 | 도로 폭 | K / 실제 설치 | Coverage | 계산 |
|---|---:|---:|---:|---|
| Library low | 7m | 3 / 3 | 99.875% | 기존 90GB / 300초 MILP 결과, best_known |
| Library high | 14m | 3 / 3 | 99.871795% | 새 90GB / 300초 MILP 결과, best_known |
| Test: 7m 단독 전이 | 10m | 3 / 3 | 99.777778% | 기하 변환 + 설치 격자 투영 |
| Test: 14m 단독 전이 | 10m | 3 / 3 | 99.814815% | 기하 변환 + 설치 격자 투영 |
| Test: 두 배치 보간 | 10m | 3 / 3 | **99.916667%** | 센서 대응 + 보간 + 설치 격자 투영 |

10m 테스트 맵 target은 10,800개입니다. 보간 배치는 10,791개를 관측하고 9개를 놓쳤습니다.
보간이 단독 전이보다 높은 것은 이번 한 사례의 결과이며 일반화 성능을 증명하지 않습니다.
10m MILP 기준값과 보간 초기화 MAPPO 보정은 **To be added**입니다. 이번 단계에서는 학습하지 않았습니다.

보간 절차:

1. 동일 K, 실제 설치 수, sensor/candidate/target 설정과 고정 기하 조건을 확인합니다.
2. 직선 맵의 대칭성과 같은 설치선(north/south)을 고려하여 Hungarian matching으로 센서를 대응시킵니다.
3. 대응 비용은 정규화된 x/z/yaw/pitch 차이이며 테스트 coverage를 사용하지 않습니다.
4. `alpha=(10-7)/(14-7)=3/7`로 x/z/pitch와 설치선 기준 횡방향 offset을 보간합니다.
5. yaw는 원형 각도 차이를 사용하고 최종 pose를 기존 XYZ/yaw/pitch 후보 격자로 투영합니다.
6. 10m 맵 mesh에서 각 배치의 실제 관측 여부를 직접 raycasting하여 평가합니다.

센서 수 또는 설치선 구성이 대응되지 않으면 실패합니다. 임의 도로/비대칭 장애물의 대칭 변환은 지원하지 않습니다.
테스트 맵에서는 전체 visibility cache나 orientation별 후보 mask를 만들지 않습니다.
설치 가능 XYZ 목록은 투영을 위해 생성하며, 각 방법당 최종 3개 pose만 raycasting합니다.
`src/core/visibility.py`의 기존 공통 mesh/FOV 경로를 재사용합니다.
두 원본 맵에서 직접 raycasting과 기존 cached evaluator의 covered mask가 완전히 같은지도 검사했습니다.

실행 명령(기존 output은 덮어쓰지 않으므로 재실행 시 새 이름 사용):

```bash
# --execute를 빼면 계획만 확인
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_offline.py \
  --road-width 14 --k-values 3 --output results/prototype_straight_w14_k3_20261002 \
  --max-edges 1000000000 --time-limit 300 --execute

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_transfer.py \
  --low results/prototype_s0_k3_90gb_20261002 \
  --high results/prototype_straight_w14_k3_20261002 \
  --width 10 --k 3 --output results/transfer_straight_w10_k3_20261002_v2 --execute

.venv/bin/python tests/test_prototype_transfer.py
```

14m 맵: target 15,600개, 원래 pose 82,800개, 동일 mask 제거 후 82,771개,
연결 495,201,852개. 설치 격자와 2m bin 규칙 때문에 폭에 따라 후보 수가 달라집니다.
새 실행의 edge 안전 상한은 명시적으로 10억으로 설정했고 후보 pruning은 하지 않았습니다.
MILP는 실제 solver 310.317초에 시간 제한(status 9)으로 종료됐으며 최적성 미인증입니다.

보간 결과 폴더:

```text
results/transfer_straight_w10_k3_20261002_v2/
  library/index.json           # 7m/14m source, hash, 인증 상태; 10m 제외
  library/low/, library/high/  # geometry/config/result/pose 및 view.html
  report.json                 # 센서 대응, alpha, 지표, 시간, 소스 hash
  test_geometry.json          # 10m 합성 맵 기하
  test_mesh.npz
  interpolation.npy / interpolation_coverage.npz / interpolation.html
  transfer_low.npy / transfer_low_coverage.npz / transfer_low.html
  transfer_high.npy / transfer_high_coverage.npz / transfer_high.html
  source/                     # 실행 시 소스 snapshot
```

[10m 보간 배치 3D](results/transfer_straight_w10_k3_20261002_v2/interpolation.html),
[7m prototype 3D](results/transfer_straight_w10_k3_20261002_v2/library/low/view.html),
[14m prototype 3D](results/transfer_straight_w10_k3_20261002_v2/library/high/view.html).
HTML은 Plotly를 포함하여 별도 서버 없이 브라우저에서 회전/확대할 수 있습니다.
표시 target은 최대 6,000개씩 샘플링하지만 수치는 전체 target 기준입니다.

`_v2`는 모바일 3D 여백을 조정한 재실행이며 MAPPO 버전명이 아닙니다.
최초 폴더도 보존되어 있고 두 실행의 coverage는 동일합니다.
기록된 세 배치 생성·맵 scene 생성·평가 합계는 최초 약 0.674초, 재실행 약 0.223초입니다.
이는 Python/import 시작, 원본 배치 검산, HTML/파일 export를 제외한 단일 실행 측정이며
전체 CLI wall time이나 정식 MILP 대비 speedup이 아닙니다.
문법 검사, 새 보간 테스트 5개, 기존 prototype 테스트 4개, pipeline 테스트 7개가 통과했습니다.

### 기존 시나리오 0 오프라인 실행 기록

`prototype_offline.py`는 기존 연구/UI 설정을 바꾸지 않는 별도 실행 경로입니다.
현재 파일럿 runner의 Gurobi `SoftMemLimit`은 사용자 요청에 따라 **90GB**입니다.
최초 K=1..6 결과는 **24GB** 실행 기록이고, K=3은 **90GB**로 재실행했습니다.
시간 제한 기본값은 300초로 유지합니다. 90GB는 Gurobi 내부 soft limit이며 Python과
행렬 등 전체 프로세스/시스템 메모리 상한이 아니므로 시스템 메모리 부족 가능성은 남습니다.
현재 지원 범위는 `--scenario-id 0/1/2` 및 `--road-width` 합성 폭 변형, 순수 union coverage, at-most-K입니다.
기존 2m XY bin·높이 3~6m·yaw 5도·pitch 2.5도 격자를 유지하고 top-M만 해제합니다.
시나리오 0의 기본 7m 맵은 전체 165,600개 pose를 평가하며 정확히 같은 coverage mask는 중복 제거합니다.
이 제거는 순수 union/at-most-K의 최적값을 보존하지만 동등한 대체 pose를 모두 보관하지는 않습니다.
연속 공간이나 인도 전체에 대한 최적화가 아닙니다.

```bash
# 계획만 확인
.venv/bin/python prototype_offline.py --k-values 1 2

# 실제 실행: 새 폴더 필수, 기존 폴더는 덮어쓰지 않음
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_offline.py \
  --output results/prototype_s0_20261002 --k-values 1 2 --time-limit 300 --execute

.venv/bin/python tests/test_prototype_offline.py
```

K=1은 전체 단일 후보 최대값을 직접 계산하고 K>=2는 GREEDY 초기해와 Gurobi를 사용합니다.
Gurobi gap 목표는 0, 현재 soft memory limit는 90GB(최초 실행 당시 24GB), sparse incidence edge 기본 상한은 250,000,000입니다.
edge 상한을 넘으면 실패로 기록하며 후보를 몰래 줄이지 않습니다.
300초는 solver 제한시간으로 후보/캐시/행렬 생성시간은 별도입니다.
`discrete_optimal`은 상태·목적값·상한 일치가 확인된 경우이고, 미인증 결과는 `best_known`입니다.
`manifest.status=complete`는 저장 작업 완료이지 모든 K의 최적성 인증 완료가 아닙니다.

2026-10-02 실제 파일럿 결과:

| K | coverage | 관측 target | 상태 |
|---:|---:|---:|---|
| 1 | 58.402778% | 4,205 / 7,200 | 전체 이산 후보에서 최적값 확인 |
| 2 | TBD | TBD | sparse 행렬 메모리 안전장치로 solver 시작 전 중단 |

전체 165,600 pose에서 동일 mask 149개를 제거해 165,451개를 저장했습니다.
candidate-target 연결 수는 421,848,646개로 안전 상한을 초과했습니다.
따라서 manifest는 `failed`이며 K=1만 완료됐습니다. 시간 초과나 라이선스 오류가 아닙니다.
K=1 pose는 `[30.25, 44.25, 3.0, 85.0, -20.0]`입니다.
위 표는 최초 실행 기록입니다. 후속 실행은 아래의 분할 행렬 경로를 사용합니다.

### K=2 이상 분할 행렬 재실행

`src/methods/milp.py`에 target별 분할 행렬 전달을 추가했습니다.
기존 수식과 모든 후보를 유지하며, 전체 COO/CSR 및 결합 행렬을 동시에 보관하지 않습니다.
파일럿은 기본 128 target씩 전달하고 기존 연구/UI 경로는 기본 행렬 방식을 유지합니다.
solver의 자체 모델/탐색 메모리는 여전히 필요하며, batch 크기가 전체 RAM 상한을 의미하지는 않습니다.
작은 문제에서 K=0..3의 비가중/가중 목적함수를 기존 행렬 방식 및 전수조사와 비교하는 회귀 테스트를 추가했습니다.

```bash
# 먼저 --execute 없이 계획 확인; 실제 사용한 명령:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_offline.py \
  --output results/prototype_s0_k2_batched_20261002 --k-values 2 \
  --reuse-candidates results/prototype_s0_20261002 \
  --max-edges 500000000 --time-limit 300 --execute

# K=3..6 후속 실행 (별도 폴더)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_offline.py \
  --output results/prototype_s0_k3_6_batched_20261002 --k-values 3 4 5 6 \
  --reuse-candidates results/prototype_s0_20261002 \
  --max-edges 500000000 --time-limit 300 --execute
```

재사용 시 config hash와 환경/후보 생성 관련 source hash를 확인합니다.
후보 archive와 cache를 새 폴더로 복사하고 SHA-256 및 원본 경로를 manifest에 남깁니다.
기존 결과를 덮어쓰지 않습니다. 위 output이 존재하면 재실행 시 새 이름을 사용합니다.
`k2/gurobi.log`는 실제 solver 로그이며 `k2/result.json`이 최종 배치와 인증 상태입니다.
새 경로의 edge 상한 5억은 이 시나리오에 명시한 실행 설정이며, 무제한 메모리 사용 허용이 아닙니다.

K=2 재실행은 정상 저장됐으며 coverage 98.666667% (7,104/7,200), 실제 설치 수 2개입니다.
solver status 9(time limit), 실제 solver 시간 377.545초, K별 wall time 464.976초입니다.
전처리 중단 지점 때문에 설정 300초를 초과했습니다. `best_bound`와 `mip_gap`의
비유한값은 JSON에서 `null`로 저장하며, Gurobi 원문 로그는 그대로 보관합니다.
이 결과는 GREEDY 초기해와 같고 `best_known`이며 최적해 인증이 아닙니다.

K=3 이후 분할 경로는 초기해의 x와 y를 모두 제공해 초기해 completion 비용을 줄입니다.
GREEDY 초기해가 target 전체를 덮으면 `ANALYTIC_FULL_COVERAGE`로 solver를 생략합니다.
이는 순수 union coverage의 이론적 상한에 도달한 배치이므로 해당 at-most-K에서 최적입니다.
다만 더 작은 K가 미인증이면 이것만으로 100% 달성에 필요한 최소 센서 수를 증명하지 못합니다.

### 시나리오 0 최신 K별 저장 결과

| 예산 K | 실제 설치 수 | coverage | 관측 / 7,200 | 인증 및 종료 사유 |
|---:|---:|---:|---:|---|
| 1 | 1 | 58.402778% | 4,205 | 전체 단일 후보 최댓값, 이산 최적 |
| 2 | 2 | 98.666667% | 7,104 | best_known, Gurobi 시간 제한(status 9) |
| 3 | 3 | 99.875000% | 7,191 | best_known, 90GB 재실행 시간 제한(status 9) |
| 4 | 4 | 100% | 7,200 | coverage 상한 달성, 이산 최적 |
| 5 | 4 | 100% | 7,200 | at-most-K, coverage 상한 달성 |
| 6 | 4 | 100% | 7,200 | at-most-K, coverage 상한 달성 |

결과 위치:

- K=1: `results/prototype_s0_20261002/k1/`
- K=2: `results/prototype_s0_k2_batched_20261002/k2/`
- K=3 최신: `results/prototype_s0_k3_90gb_20261002/k3/`
- K=3 이전 및 K=4..6: `results/prototype_s0_k3_6_batched_20261002/k{K}/`

후속 두 manifest는 `complete`입니다. 최초 실패 기록과 K=1 결과는 덮어쓰지 않았습니다.
최초 K=3 solver 시간은 279.295초이며 24GB soft memory limit에서 feasible solution을 정상 저장했습니다.
K=2와 K=3 모두 GREEDY 초기해를 개선하지 못했고 유한한 solver bound/gap을 얻지 못했습니다.
분할 방식으로 행렬 생성 메모리 복사는 줄였지만 Gurobi 자체 모델/전처리 메모리 문제까지
해결된 것은 아닙니다. 두 K의 최적해 인증은 후속 과제로 남아 있습니다.
K=4에서 100%라는 사실은 확인했으나 K=3 최적값이 미확정이므로 "최소 4대 필요"라고 단정하지 않습니다.

검증: Python 문법 검사, prototype 테스트 4개(전수조사 비교 포함), 기존 pipeline 테스트 7개,
세 방법 dry-run 통과. K=1..6 저장 pose/coverage/중복 target 수/실제 센서 수/canonical 좌표
왕복 변환을 후보 archive와 재비교하여 모두 일치했습니다. 각 run은 저장 전 공통 evaluator로도 검산합니다.
위 결과는 ignored `results/` 아래에 저장되며 저장소 clone에 자동 포함되지 않습니다.

### K=3 90GB 재실행 기록

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -u prototype_offline.py \
  --output results/prototype_s0_k3_90gb_20261002 --k-values 3 \
  --reuse-candidates results/prototype_s0_20261002 \
  --max-edges 500000000 --time-limit 300 --execute
```

결과: 실제 설치 3대, coverage 99.875% (7,191/7,200), 미관측 target 9개.
90GB 제한 실행은 메모리 제한 없이 시간 제한(status 9)으로 끝났습니다.
solver 시간 316.265초, 전체 실행 424.913초, presolve 306.51초이며
로그상 탐색 node와 simplex iteration은 모두 0입니다. GREEDY 초기해를 개선하지 못했습니다.
solver의 유한한 bound/gap을 확보하지 못해 `best_known`이며 최적해라고 해석하면 안 됩니다.
결과 저장 후 공통 evaluator coverage와 canonical 좌표 왕복 변환을 다시 검사해 통과했습니다.
manifest는 `complete`이고 이전 24GB 결과도 보존합니다.

저장 구조(각 K의 파일은 해당 계산이 완료됐을 때만 생성):
```text
results/prototype_s0_20261002/
  manifest.json        # 실행 명령, 소스 hash, 후보 통계, K별 결과
  config.yaml          # 실제 계산 설정
  geometry.json        # 지도 기하, 설치선, 출처와 가정, 좌표계
  mesh.npz             # 실제 환경 mesh
  source/              # 실행 시 소스·원본 시나리오 snapshot
  cache/               # 새로 생성한 run 전용 visibility 캐시
  candidates.npz       # 중복 제거 후 pose, packed mask, target
  k1/                  # 완료; k2/는 현재 빈 폴더
    result.json        # coverage, overlap, 미관측, solver status/bound/gap
    poses.npy
    canonical_poses.json
    validation.json
```

직선 좌표계는 중심 (50,50,0), +X 방향을 기준으로 s=x-50, d=y-50입니다.
branch는 main, 설치선은 north/south이며 yaw는 +X 기준 상대각입니다.
절대 pose와 canonical pose의 왕복 변환 및 공통 evaluator coverage 일치를 검사합니다.
geometry 출처는 OSM metadata 기반 파라메트릭 생성이며 실제 OSM 도로 polyline 복원이 아닙니다.
이 오프라인 runner 자체에는 전이·보간·자동 K 선택·학습 재개가 없습니다.
폭 전이·보간은 위의 별도 `prototype_transfer.py` 파일럿에서 수행합니다.

## 기존 전체 도로용 MAPPO v3 결과

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

- [과거 v2 세미나 보고서](results/seminar_20260919/SEMINAR_REPORT.md)
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

이하 번호 절은 기존 `main.py` Roadside GREEDY/MILP/MAPPO와 v2/v3 실험의 공통 기반입니다.
최근 source-free corridor·제약부 J·선택적 조회의 현재 상태는 [통합 명세](#current-math)를 먼저 봅니다.
여기 나오는 top-10과 2m bin은 기존 기본 YAML 경로이며 모든 실험의 후보 설정을 뜻하지 않습니다.

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

기존 세 방법의 최종 coverage는 모두 `src.environment.evaluate_coverage()`와 동일한 target, visibility cache, FOV 판정을 사용합니다.
국소 on-demand 및 최신 corridor는 공통 각도식을 사용하는 `MeshVisibility` direct 경로도 쓰므로
cache evaluator 하나만 호출하는 것은 아닙니다. 두 물리 경로의 parity 검증을 따로 유지합니다.

기존 기본 MILP/GREEDY는 위치별 상위 10개 orientation만 사용하고 MAPPO는 전체 yaw/pitch 범위를 행동으로 탐색합니다.
따라서 그 MILP는 MAPPO 전체 행동 공간에 대한 엄밀한 upper bound가 아닙니다.
10월 12일의 별도 full-grid archive 검사와 최신 K=2 전수조사는 각각 선언된 공간 안에서 별도로 해석합니다.

## 2. 현재 프로젝트 구조

아래 tree는 기존 공통 엔진과 prototype 계열의 주요 파일입니다. 후속 제약/연결/geometry-only
runner·검산기·프로토콜은 앞의 [최신 파일 책임 표](#current-reproduction)에 추가로 정리했습니다.
생성 결과의 source snapshot은 worktree와 독립이며 현재 파일 목록만으로 과거 실행을 재현하지 않습니다.

```text
hj_ws/
├── configs/
│   ├── experiment.yaml
│   ├── library_expansion.json   # 10월 8일 source 4개/target 4개 분리 명세
│   ├── shape_pilot.json         # 10월 7일 비대칭 개발 맵 4개
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
│       ├── mappo.py             # native MAPPO port와 학습 환경
│       ├── local_mappo.py       # prototype 초기화, on-demand 국소 보정
│       └── candidate_mappo.py   # UI용 후보 ID 선택형 MAPPO
├── tests/
│   ├── test_pipeline.py         # 기존 파이프라인 무결성
│   ├── test_common_placement.py # 공통 objective/회귀/가림
│   ├── test_library_expansion.py # 신규 library/전이 계획 및 축소 end-to-end
│   ├── test_shape_pilot.py      # 교차점 geometry와 기존 맵 회귀
│   ├── test_placement_ui.py     # UI API/가중치/입력 검증
│   └── test_placement_comparison.py # 복수 비교/선택/다운로드 브라우저 검증
├── placement_ui.py              # Flask 로컬 UI 서버
├── prototype_library.py         # source 전체 후보/K별 library 및 새 target 전이
├── prototype_offline.py         # K별 전체 이산 후보 library
├── prototype_compatible.py      # 설치선 배분 제약부 보조 library
├── prototype_junctions.py       # 과거 교차로 파일럿
├── prototype_transfer.py        # 도로 상대 좌표와 센서 대응/보간
├── prototype_refine.py          # 6개 배치 슬롯 및 국소 학습
├── prototype_benchmark.py       # 10월 3일 폭 변화/seed/baseline benchmark
├── prototype_segments.py        # 설치 구간 비율 좌표 보간
├── prototype_runtime.py         # 순차 계측, 초기 배치 선택/보정/기준선
├── shape_baseline_report.py     # 10월 7~8일 형상 4개 비교 집계
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

기존 전체 도로용 MAPPO v3 결과는 `results/mappo_v3_official_20260928_130911/`이며
`seminar_20260919/`는 과거 v2 기록입니다. 국소 MAPPO와 최신 library 결과는 문서 상단에서 구분합니다.
결과가 없는 clone에서는 해당 실행 명령과 설정 snapshot이 필요합니다.

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
| `prototype_library.py` | 별도 source/target suite, 전체 후보 library, K별 세 전이 슬롯, 상태/3D 저장 |
| `prototype_offline.py` | 폭 또는 explicit geometry 입력, 전체 이산 후보 archive와 K별 solver 인증 |
| `prototype_segments.py` | 같은 branch/side 구간의 u/상대 자세 보간 및 투영 |
| `prototype_runtime.py` | 새 프로세스별 시간/작업량 계측, 초기 배치 선택과 local 보정/기준선 |
| `src/methods/local_mappo.py` | 초기 pose에서 fresh PPO 국소 보정, on-demand visibility |
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

기본 설정 파일은 `configs/experiment.yaml`입니다. 이 번호 절의 세미나 비교는 과거 v2 기록이며
그 snapshot은 `results/seminar_20260919/config.yaml`입니다.
국소 prototype 실험은 원본 library의 `config.yaml`을 읽고, 각 seed 및 baseline 폴더에
실제 조건을 보존합니다. 10월 8일 library 확장은 기본 YAML을 복사한 뒤 topM=0,
solver 300초/gap 0/90GB 등을 별도 실행에 적용하며 MAPPO는 수행하지 않습니다.
기본 YAML만으로 각 결과의 top-M/solver/학습량을 판단하지 않습니다.

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

`main.py` subcommand는 milp/greedy/mappo/compare/visualize/full/roadside/robot입니다.
prototype 실험은 별도 `prototype_*.py` CLI이며 `main.py full`에 통합되지 않았습니다.
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

아래 목록은 기존 main.py/v2/v3 경로의 한계입니다. 최신 완료·미완료 판정은
[통합 현황](#current-status)과 [Gate 1~6](#proposed-model)을 기준으로 합니다.
특히 pose 전이, 공통 제약 evaluator, full-grid 후보 포함, 피드백 기여 분리는 후속 실행에서 다뤘지만
전역 가변 K·유용한 partial-visibility bound·실제 도로망 성능은 여전히 미입증입니다.

이 번호 절은 기존 순수 coverage 경로 기준입니다. 새 비용·설치 제약 보정의 지원 범위와 한계는
앞부분의 `Coverage + 비용·제약 보정 경로 (2026-10-20)`를 별도로 확인합니다.

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

확장 이력은 15개 미사용 폭의 전이·보간/국소 MAPPO, 비대칭 4개 맵의 보정/기준선,
새 비대칭 source 4개 × K1~6 library와 별도 target 4개의 전이까지 이어졌습니다.
10월 8일 library 생성 실행에서는 보간 24개 중 22개가 설치선 대응 불가였고 MAPPO는 수행하지 않았습니다.
그 뒤의 72개 국소 보정, full-grid 비교, 제약부 보정과 연결 도로/source-free 연구는 앞부분에 별도로 기록했습니다.
앞부분의 날짜별 결과를 기준으로 보며, 이 번호 절의 과거 v2/v3 실험과 합쳐
하나의 동일 조건 성능표를 만들지 않습니다.

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
