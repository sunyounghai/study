# SAHI: Slicing Aided Hyper Inference

- 저자: Fatih Cagatay Akyon, Sinan Onur Altinuc, Alptekin Temizel
- 발표: 2022 IEEE International Conference on Image Processing (ICIP 2022)
- 논문: https://arxiv.org/abs/2202.06934 (원제: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection)
- 코드: https://github.com/obss/sahi (MIT License)

## 한 줄 요약

고해상도 이미지를 겹치는 조각으로 잘라 각각 추론한 뒤 결과를 합치는 방식으로, 검출 모델을 바꾸지 않고도 작은 객체 검출 성능을 높인다.

## 해결하려는 문제

- **문제:** 드론, 위성, CCTV처럼 멀리서 촬영한 고해상도 이미지에서는 작은 객체를 잘 검출하지 못한다.
- **원인:**
    - 작은 객체는 이미지에서 차지하는 픽셀 수가 적어, 검출에 필요한 시각적 특징을 추출하기 어렵다.
    - 큰 이미지를 모델 입력 크기에 맞춰 축소하면 작은 객체의 세부 정보가 손실되거나 객체 자체가 거의 보이지 않게 된다.
    - 널리 쓰이는 검출 모델은 COCO처럼 비교적 큰 객체가 많은 데이터셋으로 학습되어 작은 객체를 충분히 학습하지 못할 수 있다.  
- **기존 방법의 한계:** 모델 구조를 변경하거나 입력 해상도를 높이는 방법은 추가적인 연산량과 메모리가 필요할 수 있으며, 기존 모델과 추론 환경에 바로 적용하기 어려울 수 있다.

## 문제 정의

- **과제:** 고해상도 이미지에서의 작은 객체 검출
- **입력:** 고해상도 이미지와 이미 학습된 일반 객체 검출 모델
- **출력:** 원본 이미지 좌표 기준의 객체 바운딩 박스와 클래스
- **조건:** 검출 모델의 구조는 바꾸지 않는다.
- **작은 객체의 기준:** COCO 평가 기준으로 면적이 32x32 픽셀보다 작은 객체 (논문의 평가 지표는 실험 섹션에서 확인)