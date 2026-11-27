# A/B LiDAR 배치 Library 연구 통합 정리와 TR-C 준비도

## Material Passport

- 기준일: 2026-11-27, Asia/Seoul.
- 상태: ANALYZED. 코드·고정 설정·완료 결과·직전 검수 기록에 근거한 연구 정리입니다.
- 직전 결과 검수: [검증 기록](results/library_lowk_cost_review_20261127_final/validation_summary.json).
- 이번 문서 작업은 새 학습, 최적화, raycasting 또는 독립 동료심사를 실행하지 않습니다.
- TR-C 준비도는 아래 근거에 따른 내부 판단이며 게재 확률·심사 결과·신규성 인증이 아닙니다.
- 원자료는 ignored 경로에 있습니다. clone만으로 결과가 포함되지는 않습니다.

## 1. 지금 연구하는 문제

> 이미 계산한 도로의 복수 배치를 도로 상대 좌표로 저장하고 새 도로의 후보로 전이하면,
> 제한된 물리 관측 계산으로 설치 제약을 만족하는 좋은 roadside LiDAR 배치를 얻을 수 있는가?

핵심은 **정답 좌표를 그대로 복사하는 것**이 아니라, 과거 해로 새 지도의 후보 집합을 만들고
새 환경에서 다시 평가·선택하는 것입니다. 현재의 최종 선택기는 제약부 MILP입니다.
RL, 검출기, 사전학습 정책은 최신 두 대형 실험에 들어 있지 않습니다.

```text
Source geometry + 물리 모델 + 설치 규칙
  -> K별 최적화 -> 최고 feasible 해 / 복수 대안 / bound / gap 저장
  -> 새 도로의 geometry로 source 검색
  -> A: 전체 topology 상대좌표 전이 / B: 구간별 후보 전이·합치기
  -> 실제 가림을 보기 전에 Q개 후보 계획 고정
  -> 새 도로에서 선택 후보만 실제 raycasting
  -> 공통 비용·설치·중요 구간 제약부 MILP
  -> 최종 배치 직접 검산
  -> 별도 전체 후보 기준선과 비교
```

사용자가 처음 제안한 세 기여의 현재 상태는 다음과 같습니다.

| 기여 후보 | 현재 도달점 | 아직 주장할 수 없는 부분 |
|---|---|---|
| topology/유형별 해를 저장하고 새 지도에 재사용 | A/B 24-source library, 복수 배치 전이, 강한 geometry-only 대비 개발셋 이점 확인 | 모든 source가 근사최적, 임의 미지 도로에서 항상 유리 |
| 대형 맵을 분해해 전체 센서 배분·필요 대수 계산 | 최대 480m 연결 도로, 지정 구간들의 후보를 합쳐 전역 at-most-K 선택 | 임의 도시 그래프의 자동 분해·교차로 결합, 도시 최소 센서 수 인증 |
| 제약을 만족하도록 재배치 | 비용·간격·pole·금지 구간·중요 구간·중첩 제약을 공통 MILP로 처리 | RL이 필요하거나 우월함, 실제 설치·통신 규정을 충족함 |

A와 B는 경쟁 알고리즘이 아니라 서로 다른 재사용 단위입니다. 두 트랙의 평균을 비교해
어느 library가 본질적으로 낫다고 결론내리지 않습니다.

## 2. 현재 실험환경

### 2.1 Source Library

2026-11-23 실행에서 구축한 source를 저센서 실험에서도 변경 없이 사용했습니다.

| 항목 | A: 전체 topology | B: 연결 도로 구성 요소 |
|---|---|---|
| 개수 | 12 | 12 |
| 형상 | T자 / 십자 | 직선 / 굽힘 corridor |
| 크기 | 정사각형 한 변 100/160m | 길이 80/120m |
| 폭 | 8/12/16m | 8/12/16m |
| 형상 조건 | 중앙 교차점, 직각 가지 | 총 굽힘 0/45도 |
| source 장면 | mixed 정적 가림 | mixed 정적 가림 |
| 저장 예산 | union K=1~8, redundancy K=4,8 | 동일 |

총 240개 K/profile 결과와 2,748개 대체 배치를 저장했습니다. 한 행마다 최대 12대안을
요청하지만 항상 12개가 생기는 것은 아닙니다. 최고값·대안 모두 실제 설치 대수를 따로 저장합니다.
source 240개 중 exact 181개, exact 포함 gap≤1% 229개라는 기존 검수 기록이 있으므로
**24개 library 전체를 인증된 근사최적 정답 모음이라고 부르면 안 됩니다.**

최신 저장물은 geometry, canonical 좌표, pose, packed visibility, target/critical 표식,
K/profile별 선택 ID·대안·목적값·상한·gap·시간곡선과 입력/코드 hash를 포함합니다.
source mask와 인증값은 source 장면에만 유효합니다. 새 도로의 mask나 성능 정답으로 복사하지 않습니다.

### 2.2 평가 지도

아래 8개 geometry에 정적 가림 단계가 결합됩니다. source와 다른 geometry이지만 반복 개발에
사용했으므로, 최신 실험에서 미사용 최종 test set이라고 부르지 않습니다.

| target | 크기/길이 | 폭 | 교차점 비율 또는 총 굽힘 | 전체 pose 후보 | target 수 | 최신 K |
|---|---:|---:|---|---:|---:|---|
| A T120 | 120×120m | 10m | (0.42, 0.55) | 1,680 | 19,206 | 2,3,4 |
| A T180 | 180×180m | 14m | (0.57, 0.43) | 2,300 | 39,288 | 2,3,4 |
| A CROSS120 | 120×120m | 10m | (0.55, 0.42) | 2,120 | 24,690 | 2,3,4 |
| A CROSS180 | 180×180m | 14m | (0.43, 0.57) | 3,160 | 54,852 | 2,3,4 |
| B160 | 160m | 10m | 15도, 2구간 | 1,480 | 4,458 | 3,4,6 |
| B240 | 240m | 14m | 30도, 3구간 | 2,240 | 9,624 | 3,4,6 |
| B320 | 320m | 10m | 60도, 4구간 | 2,980 | 9,012 | 3,4,6 |
| B480 | 480m | 14m | 75도, 6구간 | 4,460 | 19,242 | 6,8 |

B의 굽힘 각도는 전체 corridor의 값이지 구간마다 같은 각도로 반복 꺾는 설정이 아닙니다.
현재 B의 구간은 80m씩 명시적으로 나눈 station interval입니다. 실제 도로 그래프를 읽어
분해하는 기능이나 여러 교차로 유형이 섞인 네트워크는 아닙니다.

### 2.3 Grid, LiDAR, 가림

| 항목 | 최신 두 실험의 설정 |
|---|---|
| A geometry/target XY | 0.5m cell center |
| B target | station/lateral 1m sampling |
| target 높이 | 1.0/1.5/2.0m |
| 설치 XY/station | nominal 4m bin/station; A는 bin별 대표 유효 cell |
| 설치 높이 | 3/5m |
| 도로 normal 기준 yaw | -45/-22.5/0/22.5/45도 |
| pitch | -20/-10도 |
| 센서 모델 | VELODYNE64_R30: range 30m, horizontal FOV 360도, vertical FOV 26.9도 |
| 관측 계산 | 공통 각도 판정 + Open3D mesh raycasting |
| 추가 불투명체 | 4.5×1.8×1.8m, 10×2.5×3.5m, 6×0.25×2.5m box |
| 가림 단계 | clear=추가 box 없음, mixed=목록 앞 절반, dense=전체 |
| critical | A: 교차점 반경 20m; B: 구간 경계 station 전후 12m |

기존 기본 건물·도로 mesh 위에 정적 box를 더합니다. 차량/트럭/패널 크기의 대체물이지
실제 차량 움직임·교통량·객체 검출을 모사한 것은 아닙니다. 모든 가림 단계에서 dense box의
footprint 안 target/candidate를 공통으로 제외해 비교 지지집합을 유지합니다. 따라서 clear도
도로 전체 원시 target 집합과 같다고 설명하면 안 됩니다.

**0.5m target grid와 4m 설치 후보 grid는 다릅니다.** 과거 100×100m/full-0.5m 설치 격자
실험의 후보 수·인증을 현재 120~480m 실험에 가져오지 않습니다. 물리 모델은 개별 64개 beam,
반사율, point density, 날씨, calibration error를 재현하지 않습니다.

## 3. 공통 계산 수식

### 3.1 Visibility와 Coverage

지도/장면을 G, target을 t_j, 유효 pose 후보 집합을 D(G)라 둡니다.

```text
p_i = (x_i, y_i, z_i, yaw_i, pitch_i)       # roll 없음
a_ij(G) = 1[range 조건 AND FOV 조건 AND mesh LOS 조건]
```

target 상대 벡터를 (dx,dy,dz), yaw를 ψ, pitch를 θ라 하면 현재 각도 변환은 다음과 같습니다.

```text
dx_y =  dx cosψ + dy sinψ
dy_y = -dx sinψ + dy cosψ
dx_f =  dx_y cosθ + dz sinθ
dz_f = -dx_y sinθ + dz cosθ
az = atan2(dy_y, dx_f)
el = atan2(dz_f, sqrt(dx_f^2 + dy_y^2))
r  = sqrt(dx^2 + dy^2 + dz^2)
visible = (r <= 30) AND (|az| <= 180°) AND (|el| <= 13.45°)
          AND (no finite mesh hit OR first_hit >= r - 0.001m)
```

ray는 센서 원점에서 target으로 향하는 단위 벡터를 사용합니다. horizontal FOV가 360도여도
pitch로 기울인 축이 yaw에 따라 달라지므로 yaw는 vertical coverage에 영향을 줍니다.

선택 변수 x_i∈{0,1}와 density d_j에 대해:

```text
d_j(x) = sum_i a_ij x_i
y_j(x) = 1[d_j >= 1]
z_j(x) = 1[d_j >= 2]
F(x)   = sum_j y_j
C(x)   = 100 F(x) / N
O(x)   = 100 sum_j z_j / N
C_c(x) = 100 sum_(j in T_c) y_j / |T_c|
O_c(x) = 100 sum_(j in T_c) z_j / |T_c|
```

F는 관측 target 개수, C는 표본 coverage %, O는 전체 표본 중 중첩 비율입니다.
이는 연속 면적, 검출 정확도, 추적 연속성, 안전성 확률이 아닙니다. 현재 선택 목적은 무가중 F입니다.

### 3.2 최종 제약부 MILP

```text
maximize  sum_j y_j
subject to
  d_j = sum_i a_ij x_i
  y_j <= d_j <= K y_j
  sum_i x_i <= K
  sum_(i on pole s) x_i <= 1                    for every XY pole s
  x_i + x_l <= 1                               if XY distance < 4m
  sum_i c_i x_i <= B_K                         cost profile only
  sum_(j in T_c) y_j >= ceil(tau |T_c|)
  x_i,y_j binary
```

금지 구간과 물체 footprint는 D(G)를 만들 때 제거합니다. 거리와 pole 용량은 XY 기준입니다.
중첩 제약을 활성화할 때는 binary z_j 및 아래 식을 추가합니다.

```text
2 z_j <= d_j <= 1 + (K-1) z_j
sum_j z_j >= ceil(rho N)
sum_(j in T_c) z_j >= ceil(rho_c |T_c|)
```

| 실험/profile | 비용 상한 | critical 단일 coverage 하한 τ | 전체/critical 중첩 하한 |
|---|---|---:|---|
| 11-23 union | 없음 | 0 | 0 / 0 |
| 11-23 redundancy | 없음 | 0 | 0.10 / 0.30 |
| 11-24 base | 없음 | 0 | 0 / 0 |
| 11-24 cost | 1.25K | 0 | 0 / 0 |
| 11-24 cost_critical60 | 1.25K | 0.60 | 0 / 0 |
| 11-24 cost_critical80 | 1.25K | 0.80 | 0 / 0 |

**아직 비용·critical coverage·필수 중첩을 동시에 켠 대형 성능 실험은 아닙니다.**
공통 solver가 이 조합을 지원하고 작은 전수조사 테스트를 통과한 것과, 큰 성능 평가 완료는 다릅니다.

비용 모델은 다음과 같습니다.

```text
c_i = round(1 + 0.10 max(z_i-3,0) + 0.008 min_(h in H) ||xy_i-h||_2, 6)
B_K = 1.25 K
```

H는 geometry로 만든 service anchor입니다. A는 (0.15L,y_junction)/(0.85L,y_junction)에
가장 가까운 유효 pole, B는 station 0/160/320/.../종점에 가까운 pole을 사용합니다.
합성 상대 비용이며 실제 견적, cable routing, 전력/통신 연결성, 공유 시설 고정비는 아닙니다.
비용은 최소화 목적이 아닌 상한 제약입니다. 60/80% 등은 운영 규정이 아닌 고정 실험값입니다.

동일 visibility column인 target은 critical 여부까지 같을 때만 묶고 multiplicity로 가중합니다.
이 압축은 위 목적·제약을 보존합니다. 후보나 target을 임의로 버리는 pruning과 다릅니다.

### 3.3 최적성과 필요한 센서 수

동일 지도·후보·제약에서 저장 feasible 목적값을 L, 유효 상한을 U라 하면:

```text
L <= F* <= U
relative_gap = (U-L)/L                         L > 0
```

L=U이면 선언된 이산 문제의 최적 목적값입니다. 유효 gap≤1%를 현재 근사최적 인증 기준으로
사용하며, 정체나 time limit 자체는 인증이 아닙니다. 지표의 gap %와 coverage 차이 %p는 다릅니다.
source/전체 기준선은 K당 1,800초, 온라인 선택은 300초, optical surrogate는 15초입니다.
gap+정체 조기 종료는 gap≤1%, 최소 solver 시간, 120초 무개선을 함께 검사합니다.

목표 coverage η를 위한 최소 대수 문제는 후속 중심 과제가 될 수 있습니다.

```text
K_min(eta) = min { |P| : F(P) >= ceil(eta N), fixed installation constraints }
```

K대 feasible 해가 목표에 도달하면 K_min≤K입니다. 같은 제약의 K-1 문제 상한이 목표보다
낮으면 K_min≥K입니다. 두 조건을 함께 확인해야 최소 대수를 인증합니다.
비용 상한을 K와 함께 바꾸면 고정 비용 문제와 다른 문제군이 되므로 비교 정의부터 고정해야 합니다.
현재 일부 K만 실험했고 도시 전체 K_min을 구한 것이 아닙니다. 허용 최대 대수 K는 입력이며,
별도 서비스/비용 정의 없이 '최대 센서 수를 발견했다'는 연구 주장은 의미가 불명확합니다.

## 4. Library 검색·전이·후보 선별

### 4.1 검색

동일 track, A는 동일 topology인 source 중 아래 geometry 거리로 상위 3개를 고릅니다.
동률은 source 이름 순서입니다. 이 가중치는 물리 정리나 학습된 거리 함수가 아닌 고정 heuristic입니다.

```text
d_A = |w_s-w_t|/4 + |L_s-L_t|/100
d_B = |w_s-w_t|/4 + |L_s-L_t/modules_t|/80 + |angle_s-angle_t|/60
```

현재 검색에는 A의 교차점 비대칭, 실제 장애물 유사도, 비용 profile의 유사도가 직접 들어가지 않습니다.
source를 더 쌓기 전에 이 단순 검색의 실패 조건과 여러 source의 실제 기여를 분리할 필요가 있습니다.

### 4.2 A: 설치 구간 상대좌표 전이

source sensor를 branch/side, 설치 구간 비율 u, 설치선 횡편차 e, 높이, branch-relative yaw,
pitch로 표현합니다. source와 target의 해당 설치 구간을 [s_min,s_max]라 하면:

```text
u = (s_s - s_min,s)/(s_max,s - s_min,s)
s_t = s_min,t + u (s_max,t - s_min,t)
d_t = side_sign * (target_sidewalk_center_offset + e_s)
```

새 branch frame에서 XY/yaw를 복원하고 같은 branch/side/height의 실제 target 후보로 투영합니다.
가장 가까운 XY를 먼저 정하고, 같은 거리 후보에서는 원형 yaw/pitch 차이로 고릅니다.
금지 영역 때문에 raw 위치에서 더 멀리 투영될 수 있으므로 전이 자체가 성능 보장은 아닙니다.

최신 방법은 서로 다른 source의 센서를 억지로 일대일 매칭해 평균하지 않습니다.
각 source의 복수 배치를 별도 전이한 뒤 pose 후보를 모읍니다. 과거 branch/side 배분 불일치로
보간 24개 중 22개가 불가능했던 문제를 피하지만, 필요한 새로운 후보를 모두 포함한다는 보장은 없습니다.

### 4.3 B: 구간 전이와 공동 선택

target 구간 [a_m,b_m)에 source 길이 L_s의 station s_s를 옮깁니다.

```text
s_t = a_m + (s_s/L_s) (b_m-a_m)
xy_t = centerline_t(s_t) + d_t * normal_t(s_t)
```

side, 높이, inward-normal-relative yaw, pitch를 유지하고 해당 구간의 같은 설정 후보 중
station이 가장 가까운 것으로 투영합니다. A의 yaw 기준과 B의 yaw 기준은 서로 다릅니다.
여러 구간에서 나온 후보를 합쳐 **전체 K 아래에서 공동 선택**하므로 각 구간의 해를 단순 합산하지 않습니다.
실제 조립된 mesh에서 가림과 구간 간 중복을 다시 계산합니다. 지도 끝을 구/원통으로 연결하는
주기 경계는 사용하지 않으며, 끝 너머 외부 도로의 관측 기여는 모델 밖입니다.

### 4.4 비교군

| 방법 | Q개 후보를 고르는 방식 | Library 사용 |
|---|---|---|
| Library prefix | source별 대안 pose를 전이하고 source 간 순환 순서의 앞 Q개 | 예 |
| Library diverse | 전이 pool 안에서 optical union/double marginal 및 포화 후 공간 분산 | 예 |
| Geometry union | 전체 유효 후보에서 가림 없는 optical union marginal | 아니오 |
| Geometry portfolio | 동일 제약의 optical MILP 대안 pose를 모으고 union 순서로 보충 | 아니오 |
| Random | 전체 유효 후보의 seed별 비복원 무작위 Q개 | 아니오 |

모든 방법은 Q개 실제 mask를 얻은 뒤 동일 제약부 MILP로 최종 배치를 고릅니다.
따라서 최신 표는 'MILP vs RL vs greedy' 비교가 아니라 **후보 생성/선별 전략 비교**입니다.
Geometry 방법에도 실제 도로 geometry와 LiDAR range/FOV 정보가 있으며 무지한 기준선이 아닙니다.

Diverse의 optical score는 현재 0.5×새 union target 수 + 0.5×아직 두 번 관측되지 않은
target의 추가 관측 수입니다. 두 항이 0이면 기존 선택과 XY 거리가 먼 후보를 추가합니다.
이것은 좋은 실제 coverage의 보장이 아니며 실제 가림을 보지 않은 선별 heuristic입니다.

최신 요청 Q=256, 실제 Q=min(256,전이 pool 크기)로 모든 방법에 같은 수를 적용합니다.
A 실제 Q는 139/143/151/162, B는 256입니다. A에서는 prefix/diverse가 같은 후보 집합을 씁니다.
Q가 같아도 ray 수·optical 전처리·CPU 시간은 같지 않습니다. 후보 누락을 후속 LS/RL이 해결하려면
기존 Q 밖으로 움직이거나 새 query를 해야 하며, 같은 pool 안에서 최적화만 반복해 해결되지는 않습니다.

### 4.5 손실의 구분

전체 후보 D와 조회 후보 집합 R⊆D (|R|=Q)에서의 정확한 최적값을 각각 F*_D, F*_R라 하면,
feasible 반환값 L_R에 대해:

```text
F*_D - L_R = (F*_D - F*_R) + (F*_R - L_R)
             후보 누락 손실      pool 내부 탐색 손실
```

이는 문제를 진단하는 항등식이지 새로운 이론적 기여가 아닙니다. pool의 infeasible은 전체 D의
infeasible을 뜻하지 않습니다. 전체 기준선도 time-limited이면 정확한 손실 대신 다음 구간을 씁니다.

```text
100 max(0, L_D-L_R)/N <= true coverage regret in pp <= 100 (U_D-L_R)/N
```

source의 gap을 새 지도에 이식하거나 제한된 pool의 gap으로 전체 도로 근사최적성을 인증하지 않습니다.
Union의 submodularity만으로 비용·간격·중첩 하한을 함께 가진 현재 문제에 단순 greedy 보장을
적용할 수도 없습니다. 특히 최소 중첩/critical 요구는 부분집합을 제거해도 feasible인 성질을 깨뜨립니다.

## 5. 완료 실험과 결과

### 5.1 주요 이력과 보존해야 할 부정 결과

| 단계 | 결과가 말해 주는 것 | 근거 |
|---|---|---|
| 09-28 전체 MAPPO v3, 180 runs | 기존 배치 탐색 구현. 정책 전이나 최신 library 효과가 아님 | [당시 요약](results/mappo_v3_official_20260928_130911/V3_RESULTS_SUMMARY.md) |
| 10-03 폭 15맵, MAPPO 135회 | 보간의 일관된 단독 전이 우위 없음 | [폭 실험](results/prototype_width_sweep_20261003/summary.csv) |
| 10-11 동일 초기값 18개 | LS 97.642%, MAPPO 97.087%; 당시 예산에서 MAPPO 이점 없음 | [paired 결과](results/prototype_refinement_paired_20261011/paired_summary.csv) |
| 10-20~23 연결·response·prior | 후보 누락과 이동 범위가 병목; source 점수/response가 항상 geometry보다 낫지 않음 | [이력](README.md#전체-진행-이력과-결과-해석) |
| 11-09 feedback 분리, 468 workers | adaptive/no-feedback가 같은 배치를 반환. 피드백 자체의 이점 없음 | [보고서](results/geometry_mechanism_20261109/REPORT.md) |
| 11-18~19 A/B matched-query | library/union/portfolio 모두 축소 최적값 10/12 일치 | [반복 보고서](results/library_query_controls_repeat_20261119/report/REPORT.md) |
| 11-23 설치 제한×가림, 672조건 | 축소 후보에서 library 손실 1.501%p, portfolio 0.203%p. Library가 열세 | [분석](results/library_siting_large_20261123/analysis_20261123_v2/REPORT.md) |
| 11-23~24 확장 source/중첩/가림 | 더 넓은 후보와 24-source에서 library 이점 관찰 | 아래 표 |
| 11-24~25 저센서/비용/critical | 동일 제약에서 library의 전체 coverage 이점 관찰 | 아래 표 |

행 수와 worker 수를 합쳐 독립 지도 수나 학습 표본 수로 만들지 않습니다. 특히 11-23의
이전 축소 후보 실험과 이후 확장 실험은 library·domain·K·제약이 함께 달라져, 결과 반전을
'source 수를 늘린 인과 효과'라고 단정할 수 없습니다.

### 5.2 두 최신 실험의 구분

| 항목 | 11-23 중첩·가림 | 11-24 저센서·비용 |
|---|---|---|
| 상태 | 11-24 완료 | 11-25 완료, 11-27 검수 |
| 전체 경과 | 약 20시간 48분 | 약 23시간 21분 |
| source | 24개 새 구축 | 그 24개를 고정 재사용 |
| 평가 | 8 geometry×3장면×2 profile | 8 geometry×2장면×4 profile |
| K/Q | K4,8,12 / 요청 Q128,256 | 앞 지도 표의 저센서 K / 요청 Q256 |
| 온라인 행 | 2,016, 모두 feasible | 1,288 중 1,274 feasible / 14 infeasible |
| 전체 후보 기준선 | 144개 | 184개 중 182 feasible / 2 infeasible |
| 성능 질문 | 필수 중첩과 가림에서 재사용 유효성 | 고정 library가 새 비용/critical 제약에도 유효한가 |

| 방법 | 중첩·가림 coverage / 전체 K 시간 | 저센서·비용 coverage / 전체 K 시간 |
|---|---:|---:|
| Library prefix | 93.038% / 147.12초 | 78.815% / 58.12초 |
| Library diverse | 93.117% / 182.60초 | 78.954% / 94.08초 |
| Geometry union | 91.299% / 137.75초 | 74.071% / 36.18초 |
| Geometry portfolio | 92.309% / 231.45초 | 75.671% / 101.75초 |
| Random | 92.283% / 236.92초 | 76.514% / 137.27초 |

두 열은 동일 조건의 전후 성능이 아닙니다. 첫 열 품질은 288조건, 둘째는 공통 feasible
182조건 평균입니다. Random은 seed별 값을 조건 안에서 먼저 평균합니다. 시간은 각각 96/64개
case/profile/Q frontier를 한 번씩 평균하며 K별 CSV 행에 반복된 시간을 중복 가중하지 않습니다.
source 구축, process/import, 최종 검산/export는 제외됩니다. Shared-server 단일 측정입니다.

최신 diverse는 portfolio 대비 +3.283%p, 163승/0동률/19패이고 8개 geometry 평균 차이도
각각 양수입니다. 하지만 critical coverage 평균은 portfolio 89.983% > diverse 89.247%입니다.
모든 방법이 같은 182/184조건에서 feasible이므로 library만 제약을 만족한 결과는 아닙니다.
기본 배치를 비용 조건에 그대로 사용하면 방법별 약 82.6~89.1%가 비용 상한을 위반해,
비용 제약이 실제 배치 선택을 제한했음을 확인했습니다.

### 5.3 인증·실패·검수

11-24 전체 기준선 feasible 182개 중 exact 109개, exact 포함 gap≤1% 143개입니다.
39개는 1%를 넘고 최대 상대 gap은 125.952%입니다. 온라인 90행이 저장 기준선 incumbent보다
높지만 유효 상한보다 높지는 않습니다. 약한 기준선을 이긴 것이지 최적값을 능가한 것이 아닙니다.
11-23 기준선도 144개 중 exact 69개/gap≤1% 110개, 최대 gap 80.039%로 같은 한계가 있습니다.

최신 infeasible은 B320 mixed/dense, K3, cost_critical80 두 조건의 7개 방법/seed 반복입니다.
중요 target 1,989개 중 요구 1,592개에 대해 별도 유한 상한은 1,488/1,434개였습니다.
선언된 이산 후보·비용에서 불가능함을 재확인했고, 실행 실패나 0% coverage로 집계하지 않았습니다.

11-27 검수에서는 16개 case의 모든 후보 40,840개를 새 raycasting으로 검사했습니다.
제출 ray 158,998,838개, 모든 저장 mask 일치, feasible 배치 제약·CSV·hash·bound/gap 일치입니다.
관련 125 tests와 기존 CLI dry-run 3개가 통과했습니다. 같은 물리 모델의 구현 일관성 검증이지
실제 센서 정확도·정책 수렴·외부 도로 일반화를 검증한 것은 아닙니다.

## 6. TR-C라면 지금 어느 단계인가

**판단: 구현/개발 검증 단계는 상당히 진행됐고 논문 초안을 쓸 수 있지만, 지금 곧바로
TR-C 제출을 권할 정도의 방법론 차별성·외적 타당성·배치 의사결정 근거는 아직 아닙니다.**
퍼센트 완료율이나 게재 확률로 표현하지 않습니다. 과학적 기여는 코드량·실험 시간과 비례하지 않습니다.

### 6.1 저널과 선행연구에 비춰 본 위치

TR-C는 기술 자체보다 교통시스템의 계획·설계·운영과 성능에 대한 함의를 중심에 둡니다.
따라서 단순히 후보를 잘 고르는 코드가 아니라 센싱 인프라의 비용·관측 서비스·규모 확장에
어떤 의사결정 개선을 주는지가 중요하다는 것이 이 문서의 해석입니다.
[Elsevier 공식 저널 설명](https://shop.elsevier.com/journals/transportation-research-part-c-emerging-technologies/0968-090X).

관련 연구를 보면 문제의 상당 부분은 이미 다뤄졌습니다.

| 확인한 1차 문헌 | 공개 초록/소개에서 확인한 범위 | 우리 연구에 주는 경계 |
|---|---|---|
| Chen et al., TR-C 2024, 104838 | CARLA-SUMO와 검출 평가, 예산 및 recall chance constraint 기반 배치 | 'LiDAR 배치에 예산·불확실성 제약을 넣었다'만으로 신규성 주장 불가 |
| He et al., TR-C 2025, 105268 | 동적 가림과 beam 분포/인지 성능의 분석·최적화 | 기하 coverage를 실제 검출 성능과 동일시하지 말 것 |
| Roadside LiDAR Deployment Optimization..., IEEE TIM 2025 | 시공간 가림, network coverage/cost 및 pose 최적화 | '가림+비용+네트워크'라는 조합 자체를 새 기여라고 하지 말 것 |

근거: [TR-C 2024 논문](https://www.sciencedirect.com/science/article/abs/pii/S0968090X24003590),
[TR-C 2025 논문](https://www.sciencedirect.com/science/article/pii/S0968090X25002724),
[TIM 2025 논문](https://ieeexplore.ieee.org/abstract/document/10950411).
2026-11-27에 publisher 검색 결과의 초록/소개와 접근 가능한 저널 설명을 확인했습니다.
일부 본문 URL은 접근 제한이 있어 논문 전문을 검토한 체계적 novelty 조사로 간주하지 않습니다.
위 연구와 동일 task·domain에서 직접 성능 비교한 것도 아닙니다.

**검출기를 지금 추가해야만 TR-C가 되는 것은 아닙니다.** 이는 저널의 필수 요건이라는 주장이
아니라 범위 설정에 대한 판단입니다. 검출을 제외한다면 contribution을 센싱 인프라의 기하학적
서비스 설계로 제한하고, 실제 도로 형상·허용 사이트·경로별 관측 공백·설치비용 등으로 교통적
의미를 입증해야 합니다. 기하 관측이 좋아졌다는 사실만으로 사고 감소·검출률 향상을 주장하지 않습니다.

### 6.2 준비도 점검

| 평가 축 | 현재 상태 | 투고 전에 필요한 증거 |
|---|---|---|
| 문제 정의·공통 물리/제약 | 구현 및 수치 검수 완료 | 실제 서비스 요구와 proxy의 연결, 선택한 단순화의 민감도 |
| Library/전이/공동 선택 | A/B 작동, 24-source 및 480m까지 평가 | 이 표현/검색/대안 구성의 필수성, 실패 범위와 차별성 |
| 비교 실험 | 강한 geometry-only·random·전체 후보 비교 완료 | 제약부 greedy+swap/LS, 동일 품질 시간 비교, 약한 기준선 보강 |
| 일반화·통계 | 개발 8 geometry에서 기술통계 확보 | 사전에 봉인한 geometry family·장애물 seeds, map 단위 불확실성 |
| 계산 효율 | 준비/선별/query/solver 시간과 rays 기록 | 반복 cold-start, method 순서 통제, offline 상각, scaling |
| 실제 도로망 의사결정 | 구간 공동 K 선택까지 부분 구현 | 비정형 도로/교차로 네트워크·실제 또는 근거 있는 허용 사이트·목표 서비스 필요 대수 |
| 신규성 | 후보 기여는 명확해짐 | 관련 transfer/amortized/compositional sensor placement 문헌과 직접 차별 검증 |
| 원고·재현 패키지 | 코드·로그·명세·수치 검증 기반 확보 | 일관된 원고/그림, frozen test protocol, 배포 가능한 축소 데이터·환경 |

최대 약점은 '학습량 부족'이 아닙니다. **후보 집합의 재사용이 왜 유효한지, 언제 실패하는지,
어떤 실제 인프라 설계 판단을 개선하는지**가 아직 충분히 분리·검증되지 않았습니다.

## 7. 다음 실험의 우선순위

아래는 제안이며 이번 정리가 실행을 승인하거나 새 실험을 시작하지는 않습니다.

1. **기여와 protocol 고정:** 라이브러리 재사용을 주기여로 두고 RL·검출은 보류합니다.
   관련 연구와 비교해 재사용 단위/경계 표현/선별 규칙 중 새로운 부분을 하나로 좁힙니다.
2. **기준선·품질/시간 보강:** gap이 큰 39조건의 원인(incumbent/상한/메모리)을 분리합니다.
   독립 제약부 greedy+swap/LS를 추가하고, 전체와 shortlist 방법의 같은 시간/같은 품질 비교를 합니다.
   온라인 해를 기준선에 주는 별도 best-known 분석은 정보 경계와 비용을 밝히고 독립 기준선과 구분합니다.
3. **재사용 효과 ablation:** 고정 target/후보/물리 query 예산에서 source 1/3/다수,
   incumbent-only/대안 포함, 근접 source/무작위 source, 전이 prefix/diverse/geometry를 분리합니다.
   library 크기에 따라 Q가 함께 바뀌지 않도록 실험 설계를 고정합니다.
4. **미사용 평가:** source/train-development-test family를 분리하고 검색 가중치 등을 봉인합니다.
   교차각·곡률·비대칭 가지·폭·가림 seed를 독립적으로 바꿉니다. 필요한 지도 수는 예산과 분산을
   보고 정하며 K/profile 행을 독립 표본처럼 사용하는 검정을 피합니다.
5. **교통 인프라 사례와 대수 결정:** 실제 도로 GIS/mesh 또는 근거 있는 비정형 도로망,
   허용 pole/금지 구간/중요 연결부를 사용합니다. 목표 coverage/critical 서비스에서 필요한 대수와
   비용의 feasible 해 및 하한을 보고합니다. 현장 설치가 없으면 그 범위도 명시합니다.
6. **오프라인 비용과 경계:** 여러 지도 재사용 시의 상각과 library miss 시 fallback을 측정합니다.
   소수 sensor 고장 또는 복수 가림 상태에 동일 배치를 요구하는 강건성은 별도 확장으로 검토합니다.

동일 품질을 만족한다는 전제에서만 다음 상각식을 해석할 수 있습니다.

```text
T_reuse(n) = T_library + sum_(t=1..n) T_online,t
T_fresh(n) = sum_(t=1..n) T_full,t
n_break_even >= T_library / (mean(T_full) - mean(T_online))   denominator > 0
```

현재 숫자로 이 손익분기점을 확정하지 않았습니다. Library 업데이트 비용·저장 비용·실패 후 재계산도
후속에는 포함해야 합니다. 검출·RL·제약 종류를 동시에 늘리면 원인 분리가 어려워집니다.

### 후속 구현: 고정 Q 원인 분리

이 정리 이후 [LIBRARY_MECHANISM_PROTOCOL.md](LIBRARY_MECHANISM_PROTOCOL.md)와
`library_mechanism_study.py`를 추가했습니다. 위 3번의 source 1/3/6개, 최고 배치만/대체 배치 포함,
근접/무작위 source, prefix/diverse/geometry 비교를 **실제 Q=256 고정**으로 구성합니다.
후보 부족분은 geometry union으로 보충하고 그 수를 공개하므로 순수 library가 아닌 hybrid입니다.
기존 개발 지도만 사용하며, 별도로 예약한 geometry 8개는 이번 실행에서 차단합니다.
정식 비교 결과와 최종 미사용 평가 결과는 아직 TBD입니다. 이 추가가 위 완료 실험의 수치를
바꾸거나 source 수 증가의 효과를 이미 입증한 것은 아닙니다. 축소 검증 범위는 새 명세를 봅니다.

## 8. 지금 쓸 수 있는 논문 구조

가제: **Reusable Deployment Libraries for Geometry- and Constraint-Aware Roadside LiDAR Placement**.
가제는 제안일 뿐 novelty가 확인된 알고리즘 이름이 아닙니다.

1. 교통 인프라 배치 문제와 재사용의 필요성, 기존 sensor placement/transfer 연구와의 차이.
2. 3D 기하 관측·이산 후보·예산/설치/서비스 제약의 문제 정의.
3. A/B library 표현, source 검색, 상대좌표 전이, 예산 내 후보 선별과 공동 선택.
4. 후보 누락/solver 손실의 분리, feasible 검산, 최적성 인증 범위.
5. 개발 실험과 부정 결과, frozen held-out 비교, scaling 및 비용 상각.
6. 실제 또는 근거 있는 도로망 사례, 필요한 대수/비용/서비스 trade-off.
7. 한계: 검출 미포함, 이산화, static occlusion, 공유 서버 계측, 운영 기준 미확정.

지금은 2~4절의 구현 명세와 5절의 개발 결과를 쓸 수 있습니다. 1절의 차별성 근거,
5절의 최종 독립 검증, 6절의 교통 의사결정 증거가 핵심 미완료 부분입니다.

## 9. 코드·결과를 읽는 순서

| 파일 | 역할 |
|---|---|
| [library_rsu_stress.py](library_rsu_stress.py) | source24 구축, library pool/선별, 중첩·가림 실험 |
| [src/roadside/rsu_stress.py](src/roadside/rsu_stress.py) | 실제 domain, box, source 검색, A/B 전이 |
| [prototype_segments.py](prototype_segments.py) | A 설치 구간 상대좌표 |
| [src/methods/rsu_coverage_milp.py](src/methods/rsu_coverage_milp.py) | 공통 제약부 MILP, 압축, bound/frontier |
| [library_budget_study.py](library_budget_study.py) | 고정 source를 사용하는 저센서·비용·critical 실험 |
| [library_budget_report.py](library_budget_report.py) | 완료 결과 재검산/집계, 학습·최적화 없음 |
| [최신 보고서](results/library_lowk_cost_review_20261127_final/REPORT.md) | 방법별 성능과 검수 범위 |
| [이전 중첩 실험 manifest](results/library_rsu_stress_20261123/manifest.json) | 실제 실행 완료/provenance |
| [최신 manifest](results/library_lowk_cost_20261124/manifest.json) | 16 case, 1,288행의 실제 실행 정보 |

원 실행은 각 manifest의 command 및 [중첩 명세](LIBRARY_RSU_STRESS_PROTOCOL.md),
[비용 명세](LIBRARY_LOWK_COST_PROTOCOL.md)를 사용합니다. 소스 변경 후 과거 결과의 frozen source와
현재 checkout을 혼동하지 않습니다. 읽기 전용 계획 및 검수 형식은 다음과 같습니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python library_budget_study.py \
  --spec configs/library_lowk_cost_20261124.json

# 새 output만 허용. --direct는 실제 target 후보 재-raycasting이며 최적화/학습이 아님.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -B library_budget_report.py \
  --root results/library_lowk_cost_20261124 \
  --output results/library_lowk_cost_review_repeat --direct
```

통계 해석은 직전 보고서의 11가지 오류 점검을 유지합니다. 개발 맵 재사용, feasible 조건 선택,
map/K/seed 의존성, 사후 분석과 조건별 역전 가능성을 공개하며 p-value나 신뢰구간을 새로 만들지 않았습니다.
