# TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications

- 저자: Yu-Chuan Huang, I-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, Wen-Chih Peng
- 발표: AVSS 2019
- 논문: https://arxiv.org/abs/1907.03698
- 코드: 공식 V1 코드 확인 안 됨

## 한 줄 요약
스포츠 중계 영상에서 작고 빠르게 움직이는 공을 추적하기 어려운 문제를 해결하기 위해, 연속된 프레임을 입력받아 공의 위치를 히트맵(Heatmap)으로 예측하는 방법을 제안한다.

## 해결하려는 문제

- **배경:** 공의 궤적 데이터는 선수 평가와 경기 전략 분석에 중요한 기초 정보다.
- **문제:** 공은 크기가 작고 이동 속도가 매우 빨라 영상에서 흐릿하게 촬영되거나 정확한 위치를 파악하기 어렵다.
- **대상 영상:**
    - 주요 대상은 테니스 중계 영상이다. 공이 작고 흐리며, 잔상(afterimage)이 남거나 아예 보이지 않는 경우도 있다.
    - 확장 실험으로 배드민턴도 다룬다.
    - 방송 영상뿐 아니라 스마트폰 같은 일반 기기로 찍은 영상도 대상으로 말한다.

