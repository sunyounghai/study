# WASB: Widely Applicable Strong Baseline for Sports Ball Detection and Tracking

- 저자: Shuhei Tarashima, Muhammad Abdul Haq, Yushan Wang, Norio Tagawa
- 발표: BMVC 2023
- 논문: https://arxiv.org/abs/2311.05237
- 코드: https://github.com/nttcom/WASB-SBDT (MIT License)


## 한 줄 요약

기존 SBDT(Sports Ball Detection and Tracking) 방법들이 특정 종목에 특화되어 있거나 작은 공의 검출 및 시간적 일관성에 한계를 보이는 문제를 해결하기 위해 **고해상도 특징 추출 + 위치 인지 학습 + temporal consistency를 고려한 inference**를 결합한 범용적인 강력한 baseline인 WASB를 제안했고, 5개 종목 데이터셋에서 기존 방법들과 비교하여 우수한 성능을 보였다.

## 해결하려는 문제

- 기존 SBDT 연구들은 특정 스포츠에 맞춰 설계된 경우가 많아 여러 스포츠에 일반적으로 적용하기 어렵다.
- 전통적인 배경 차분 기반 방법은 선수 등 공이 아닌 움직이는 물체를 공으로 오검출하기 쉽다.
- CNN 기반 방법이 성능을 크게 개선했지만 여전히 다음과 같은 문제가 있다.
    - **작은 공 검출 문제:** 스포츠 공은 영상에서 매우 작기 때문에 정확한 위치를 검출하기 어렵고, 공 픽셀과 배경 픽셀의 비율이 극단적으로 불균형하다. 기존 연구는 focal loss, combo loss, hard negative mining 등의 방법을 사용해 왔으며 작은 객체를 표현하려면 고해상도 feature가 중요하다.    
    - **Temporal consistency 문제:** 기존 방법은 몇 프레임씩 묶어서 공의 움직임을 보지만, 프레임 묶음 간의 공 위치를 지속적으로 추적하지 않는다. 따라서 앞 구간에서 찾은 공의 위치와 비교해 현재 검출 결과가 자연스럽게 이어지는지 확인하기 어렵고, 공 위치가 갑자기 엉뚱한 곳으로 튀는 오검출이 발생할 수 있다.

## 문제 정의

- **Input:** 연속된 N개 프레임을 채널 방향으로 이어 붙인 텐서 HxWx3N (3.1절)
    - 실험 설정은 N=3, 각 프레임을 288x512로 resize → 288x512x9 (5.2절)
    - 원본 데이터셋 해상도는 HD/FHD 범위 (Table 1)

- **Output:** 입력과 같은 spatial resolution의 heatmap N장, HxWxN (3.1절)
    - N=3이므로 288x512 heatmap 3장 (5.2절)
    - 입력 프레임 하나당 heatmap 하나씩 대응하는 MIMO 구조
        - 단, 각 heatmap은 입력된 N개 프레임의 정보를 함께 사용해 생성됨
    - 모델 출력은 heatmap까지이며, 공 좌표는 inference 단계의 후처리로 계산
    - heatmap → 0.5 threshold → blob 검출 → 각 blob의 위치·confidence 계산 → 후보 선택
    - WASB의 CoH에서는 blob 내부의 heatmap 값을 가중치로 사용해 weighted center를 공 위치로 계산하고, heatmap 값의 합을 confidence로 사용
    - 결과: 프레임당 공 좌표 (x,y) 최대 1개

- **Task:** 각 프레임에서 공의 (x,y) 좌표를 검출하고, inference 단계에서 temporal consistency를 고려해 연속적인 ball trajectory를 얻음
    - **추적 방식: Online Tracking (3.3절)**
        - 직전 3프레임(t, t-1, t-2)의 공 위치로 local motion model(등가속도, 식 4)을 이용해 t+1 위치 예측
        - 예측 위치와 너무 먼 detection candidate를 제거
        - 남은 후보 중 confidence가 가장 높은 후보를 선택
        - Kalman filter와 particle filter는 성능 향상이 없어 사용하지 않음
    - Step=3(3장씩 겹치지 않게 묶어 처리) 기준 5개 데이터셋 중 4개에서 30 FPS 이상으로, 실시간 처리 가능한 수준 (5.3절 / Table 2)

- **Assumption / Limitation (5.5절)**
    - 프레임당 공은 최대 1개 (여러 공을 동시에 사용하는 종목에는 적용 불가)
    - 고정 카메라와 카메라가 움직이는 영상 모두 적용 가능
    - 해상도와 FPS에 이론적 제한은 없지만, 검증 범위는 HD/FHD 및 25~30 FPS
    - 공 위치를 bounding box가 아닌 중심점 (x,y)으로 표현 (3장 각주)

## 핵심 아이디어

### 1. 고해상도 특징 추출 (High-Resolution Feature Extraction, 3.1절)

- **기존 한계:**
    - 기존 방법(DeepBall, TrackNet 등)의 encoder-decoder 방식은 의미 정보가 풍부하지만 spatial resolution이 낮은 decoder feature와 이를 보완하기 위한 encoder intermediate feature를 결합해 heatmap을 만든다.
    - 그러나 결합되는 feature들이 각각 **높은 spatial resolution과 풍부한 semantic information을 동시에 충분히 갖지 못한다**는 한계가 있다.
    - 스포츠 공처럼 매우 작은 객체를 정확히 검출하려면 두 정보를 모두 갖는 feature representation이 중요하다.

- **WASB의 방법:**
    - **HRNet의 고해상도 feature extraction 방식** 사용: 여러 해상도의 branch를 병렬로 유지하고 정보를 반복적으로 교환해, spatial resolution을 유지하면서 semantic information이 풍부한 feature를 얻음 (Figure 2)
    - small HRNet 설계를 따름(경량 HRNet, 파라미터 약 1.5M)
    - 4개 stage로 구성: stage가 진행될수록 더 낮은 해상도의 branch를 하나씩 추가하고, 각 stage에서 해상도 간 정보를 교환하는 multi-resolution fusion 수행
        - 고해상도 branch: 세밀한 spatial information 유지
        - 저해상도 branch: 넓은 영역의 semantic information 확보
    - 원래 HRNet은 stem에서 입력 해상도를 1/4로 줄이지만, WASB는 stem의 stride를 제거해 더 높은 해상도의 feature를 HRMs(High-Resolution Modules)에 전달 (Figure 3)
    - 대신 stride 제거로 계산량이 증가하며 inference 속도가 감소함
    - WASB는 성능과 효율의 균형을 고려해 Figure 3(c)를 기본 설정으로 사용

- **효과 (Table 3, 축구 기준):**
    - stem 구조 (a) → (b) → (c)
        - F1: 81.7 → 86.4 → 88.3
        - AP: 71.7 → 79.0 → 83.6
        - FPS: 85.7 → 76.7 → 55.7
    - 즉, intermediate feature의 spatial resolution을 높일수록 SBDT 성능은 좋아지지만 inference 속도는 감소함
    - 다른 제안 기법을 추가하지 않은 상태에서도 기존 최고 방법(TrackNetV2, F1 86.6)을 넘어 F1 87.3을 달성 (Table 4)

### 2. 위치 인지 학습 (Position-Aware Model Training, 3.2절)

- **기존 한계:** 기존 방법은 정답 공 위치에서 거리 d 이내의 픽셀을 모두 1, 나머지를 0으로 만드는 binary GT map(식 1)으로 학습한다.
    - 정답 위치 주변(반경 d)의 픽셀이 모두 같은 값 1을 가지므로, **정확한 공 위치 정보가 모호해지고 모델이 exact ball position에 덜 민감해진다.**

- **WASB의 방법:** 
    - 중심에서 멀어질수록 값이 작아지는 **real-valued GT map**을 사용해 공의 정확한 위치 정보를 더 세밀하게 표현 (식 2, Figure 4)
        - 실험 설정: d=2.5, c_min=0.7 → 반경 안의 값이 가장자리 0.7 ~ 중심 1 (5.2절)
    - **Quality Focal Loss**로 학습 (식 3)
        - GT가 binary인 경우에는 기존 focal loss와 동일
    - **HLSM (Hard-to-Localize Sample Mining):**
        - real-valued GT를 모든 학습 데이터에 적용해도 통계적으로 성능 향상이 없었기 때문에 **위치를 찾기 어려운 샘플에만 적용**
        - 학습 중 전체 training sequence를 inference하여 예측 위치가 GT 위치에서 먼 이미지를 hard-to-localize sample로 선정
        - 선정된 이미지의 GT를 real-valued GT로 바꾸고 남은 epoch 동안 추가 학습
        - 실험에서는 총 30 epoch 중 **epoch 20 시작 시점에 HLSM을 한 번 수행** (5.2절)
        - 복잡한 배경 때문에 흐릿했던 heatmap이 더 선명해지고, 공 위치를 더 정확하게 찾을 수 있음 (Figure 5)

- **효과 (Table 4, 축구 기준):**
    - HLSM 추가 시 F1: 87.3 → 87.8
    - AP: 80.1 → 81.1
    - 학습 단계에서만 적용되므로 inference 속도에는 영향 없음

### 3. 시간적 일관성을 고려한 추론 (Inference, 3.3절)

- **기존 한계:** 기본 inference는 각 heatmap에서 찾은 blob의 기하학적 중심을 공 위치로, blob 크기를 confidence로 사용하고, 해당 이미지 안에서 confidence가 가장 높은 후보를 선택한다.
    - 따라서 공과 비슷한 물체가 함께 검출되면 시간적 정보 없이 현재 이미지의 confidence만으로 잘못된 후보를 선택할 수 있다.

- **WASB의 방법:**
    - **CoH (Center of Heatmap):** blob 내부의 heatmap 값을 가중치로 사용해 weighted center를 공 위치로 계산하고, heatmap 값의 합을 confidence로 사용
    - **Online Tracking:** 이전 프레임들의 공 위치로 현재 공의 예상 위치를 계산하고, 예상 위치에서 너무 먼 detection candidate를 제거한 뒤 남은 후보 중 confidence가 가장 높은 후보를 선택
        - Temporal information은 새로운 위치를 직접 생성하기보다는 **일관되지 않은 detection candidate를 필터링하는 데 사용**
        - Kalman filter와 particle filter는 성능 향상이 없어 사용하지 않음
    - **Oversampling:** 같은 이미지를 서로 다른 MIMO 프레임 조합에 포함시켜 다양한 detection candidate를 얻고, 이를 모두 다음 후보 선택 단계에서 활용
        - Step=1에서는 입력 프레임 조합을 한 프레임씩 이동시키며 oversampling

- **효과 (Table 4):**
    - CoH (축구): F1 87.8 → 88.3, AP 81.1 → 83.6
    - Online Tracking:
        - Soccer: F1 88.3 → 88.3, AP 83.6 → 83.6
        - Tennis: F1 93.9 → 94.0, AP 90.8 → 91.0
        - Badminton: F1 91.6 → 91.6, AP 88.5 → 88.5
    - Step=1 (축구): AP 83.6 → 86.2로 향상되지만 F1은 88.3 → 88.2로 거의 변화 없고, FPS는 55.7 → 23.6으로 감소 (Table 2)


## 실험

### 데이터셋 (4.1절, Table 1)

5개 스포츠의 공 검출/추적 데이터셋을 사용했다.

- 새로 도입한 데이터셋: 배구, 농구 (SBDT 분야에서 처음 사용)
- 새로 라벨링한 데이터셋: 축구, 농구 (프레임별 공 위치를 수동으로 라벨링)

| 종목 | 해상도 | FPS | 학습 프레임 | 테스트 프레임 | 평균 공 이동 거리 (disp.) |
|---|---|---:|---:|---:|---:|
| 축구 | 1920×1080 | 25 | 11,994 | 5,999 | 10.4 / 15.7 px |
| 테니스 | 1280×720 | 30 | 14,160 | 5,675 | 15.3 / 13.6 px |
| 배드민턴 | 1280×720 | 30 | 78,558 | 12,656 | 11.8 / 12.5 px |
| 배구 | 1280×720 | N/A | 143,213 | 54,817 | 14.4 / 15.1 px |
| 농구 | 1920×1080 | N/A | 244,224 | 31,104 | 33.7 / 33.9 px |

- **disp.**: 연속된 두 프레임 사이에서 공이 평균적으로 이동한 픽셀 거리
  - 학습 / 테스트 순서
- 해상도는 각 데이터셋에서 가장 많은 영상의 해상도
- 배구·농구 데이터셋은 원본 FPS 정보가 없음

### 데이터셋별 특징

#### 축구
- 기존 공·선수 추적을 위해 만들어진 6개의 동기화 영상 사용
- 앞 4개 클립 → 학습 / 나머지 2개 클립 → 테스트
- 기존 공 위치 라벨의 정확도가 부족하여 모든 프레임을 다시 라벨링
- 학습·테스트 모두 1경기로 표시됨 → 같은 경기의 다른 카메라 영상으로 나눈 것으로 보임 (Table 1 기준 추론)

#### 농구
- 라벨링된 이미지 275,328장으로, 당시 SBDT 분야에서 가장 큰 데이터셋
- 5개 종목 중 평균 공 이동 거리가 가장 큼
- 카메라 움직임과 빠른 줌이 자주 발생하여 공의 움직임이 복잡함

#### 배구
- 전체 클립의 약 3.7%에서는 공이 등장하지 않음

### 데이터셋을 보면서 알게 된 점

- 종목마다 공의 이동 속도와 영상 촬영 방식이 크게 다름
- 특히 농구는 큰 `disp.`와 카메라 움직임 때문에 빠른 공 움직임과 움직이는 카메라에 대한 대응력을 평가하기에 적합해 보임
- 축구 데이터셋은 학습/테스트가 같은 경기 기반으로 보이므로 **다른 경기나 경기장에서의 일반화 성능은 별도로 검증할 필요가 있음**