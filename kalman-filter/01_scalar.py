"""
스칼라 칼만 필터 (숫자 하나를 추정)
"""

import numpy as np

# 아래 숫자 바꿔서 테스트
TRUE_VALUE = 10.0 # 진짜 값 (원래 모른다고 가정)
R_STD = 2.0 # 측정 노이즈 표준편차 (클수록 측정값이 넓게 흩어짐)
Q = 0.0 # 진짜 값이 변할 수 있는 정도 (0: 값은 고정이라고 가정)
INIT_X = 0.0 # 처음 추정값
INIT_P = 100.0 # 처음 추정에 대한 불확실성
N_STEPS = 30 # 측정 횟수
SEED = 0 # 난수 시드
JUMP_AT = None # None이면 값이 안 변함

# 분산
R = R_STD ** 2  

# 가짜 측정 데이터 생성
rng = np.random.default_rng(SEED)
true_values = np.full(N_STEPS, TRUE_VALUE)
if JUMP_AT is not None:
    true_values[JUMP_AT - 1:] += 5.0
measurements = true_values + rng.normal(0, R_STD, N_STEPS)

# 칼만 필터
x, P = INIT_X, INIT_P
est_list, K_list = [], []

print(f"{'step':>4} {'측정값 z':>9} {'이득 K':>8} {'추정값 x':>9} {'불확실성 P':>11} {'running mean':>13}")
for k, z in enumerate(measurements, start=1):
    P = P + Q               # 1) 예측
    K = P / (P + R)         # 2) 이득
    x = x + K * (z - x)     # 3) 갱신
    P = (1 - K) * P         # 4) 불확실성 줄이기

    est_list.append(x)
    K_list.append(K)
    running_mean = measurements[:k].mean() # 비교용 (지금까지 측정값의 단순 평균)
    print(f"{k:>4} {z:>9.2f} {K:>8.3f} {x:>9.2f} {P:>11.3f} {running_mean:>13.2f}")
