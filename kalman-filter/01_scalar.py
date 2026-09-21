"""
스칼라 칼만 필터 (숫자 하나를 추정)

문제)
노이즈가 낀 센서 값 여러 개를 보고 실제 값이 얼마인지 추정을 하고 싶음
ex) 가만히 서 있는 선수의 실제 x의 위치가 10m이지만 
영상에서 측정한 값은 매번 10.3, 8.1, 11.7 .. 처럼 흔들림
"""

import numpy as np
import matplotlib.pyplot as plt

# 아래 숫자 바꿔서 테스트
TRUE_VALUE = 10.0 # 진짜 값 (원래 모른다고 가정)
R_STD = 2.0 # 측정 노이즈 표준편차 (클수록 측정값이 넓게 흩어짐)
Q = 0.0 # 진짜 값이 변할 수 있는 정도 (0: 값은 고정이라고 가정)
INIT_X = 0.0 # 처음 추정값
INIT_P = 100.0 # 처음 추정에 대한 불확실성 (클수록 처음 추정을 못 믿음)
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

# RMS 오차 (실제 값과의 차이)
est_arr = np.array(est_list)
print(f"\n측정값 RMS 오차: {np.sqrt(np.mean((measurements - true_values) ** 2)):.2f}")
print(f"추정값 RMS 오차: {np.sqrt(np.mean((est_arr - true_values) ** 2)):.2f}")

# 그래프
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
steps = np.arange(1, N_STEPS + 1)

axes[0].plot(steps, true_values, "k-", lw=2, label="true value")
axes[0].plot(steps, measurements, "o", color="tab:red", alpha=0.5, label="measurement z")
axes[0].plot(steps, est_list, "-", color="tab:blue", lw=2, label="Kalman estimate x")
axes[0].set_xlabel("step")
axes[0].set_ylabel("value")
axes[0].set_title("Estimate vs measurement")
axes[0].legend()

axes[1].plot(steps, K_list, "o-", color="tab:green")
axes[1].set_ylim(0, 1.05)
axes[1].set_xlabel("step")
axes[1].set_ylabel("Kalman gain K")
axes[1].set_title("K = how much we trust the new measurement")

fig.tight_layout()
fig.savefig(f"results/01_Q{Q}_R{R_STD}_jump{JUMP_AT}_seed{SEED}.png", dpi=120)