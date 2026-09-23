"""
1차원 위치+속도 칼만 필터
"""
import numpy as np

# 아래 숫자 바꿔서 테스트
DT = 0.1 # 측정 간격(초)
N_STEPS = 40 # 측정 횟수
TRUE_VEL = 5.0 # 선수의 진짜 속도(m/s), 처음엔 등속
R_STD = 1.0 # 위치 측정 노이즈 표준편차 (m), 클수록 측정을 못 믿음
SIGMA_A = 1.0
CHANGE_AT = None # None이면 계속 등속, 숫자 입력 시 그 스텝부터 NEW_VEL로 바뀜
NEW_VEL = 0.0 # CHANGE_AT 이후의 진짜 속도
PRINT_EVERY = 2 # 표를 몇 스텝마다 출력할지
SEED = 0

# 가짜 데이터 생성
rng = np.random.default_rng(SEED)
true_vel = np.full(N_STEPS, TRUE_VEL)
if CHANGE_AT is not None:
    true_vel[CHANGE_AT - 1:] = NEW_VEL
true_pos = np.cumsum(true_vel) * DT # 속도를 누적해서 위치를 만듦
measured_pos = true_pos + rng.normal(0, R_STD, N_STEPS) # 위치에만 노이즈를 얹음

# 칼만 필터 세팅
F = np.array([[1.0, DT],
              [0.0, 1.0]])
H = np.array([[1.0, 0.0]])
R = np.array([[R_STD ** 2]])
Q = SIGMA_A ** 2 * np.array([[DT ** 4 / 4, DT ** 3 / 2],
                             [DT ** 3 / 2, DT ** 2]])

# 초기 상태: 위치는 첫 측정값, 속도는 모르니까 0, 속도의 불확실성은 아주 크게 줌
x = np.array([[measured_pos[0]],
              [0.0]])
P = np.diag([R_STD ** 2, 10.0 ** 2])

est_pos = [x[0, 0]]
est_vel = [x[1, 0]]
naive_vel = [np.nan]
Ks = [(np.nan, np.nan)]

print(f"{'step':>4} {'측정 위치':>9} {'위치 추정':>9} {'속도 추정':>9} {'진짜 속도':>9} "
      f"{'미분 속도':>9} {'K_위치':>7} {'K_속도':>7}")

for k in range(1, N_STEPS):
    z = np.array([[measured_pos[k]]])

    # 1) 예측
    x = F @ x
    P = F @ P @ F.T + Q

    # 2) 이득 (K는 2x1 벡터: [위치용 이득, 속도용 이득])
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    # 3) 갱신: (측정 - 예측)의 차이를 K만큼 반영
    x = x + K @ (z - H @ x)
    P = (np.eye(2) - K @ H) @ P

    est_pos.append(x[0, 0])
    est_vel.append(x[1, 0])
    naive = (measured_pos[k] - measured_pos[k - 1]) / DT
    naive_vel.append(naive)
    Ks.append((K[0, 0], K[1, 0]))

    if k % PRINT_EVERY == 0:
        print(f"{k:>4} {measured_pos[k]:>9.2f} {x[0, 0]:>9.2f} {x[1, 0]:>9.2f} "
              f"{true_vel[k]:>9.2f} {naive:>9.2f} {K[0, 0]:>7.3f} {K[1, 0]:>7.3f}")

# 요약 숫자
est_vel_arr = np.array(est_vel)
naive_arr = np.array(naive_vel)
rmse_kf = np.sqrt(np.mean((est_vel_arr[5:] - true_vel[5:]) ** 2))
rmse_naive = np.sqrt(np.nanmean((naive_arr[5:] - true_vel[5:]) ** 2))
print(f"\n속도 오차(RMSE, 처음 5스텝 제외): 미분 {rmse_naive:.2f} m/s   vs   칼만 {rmse_kf:.2f} m/s")

