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