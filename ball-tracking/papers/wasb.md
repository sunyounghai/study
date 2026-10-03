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
