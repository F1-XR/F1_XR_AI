"""추월 예측 모델 추론 모듈.

학습은 별도 레포(f1-overtake-pipeline)에서 하고, 여기선 완성된 모델(.txt)만 로드해 '예측'만 한다.
  - features.py : 단일시점 26피처 빌더 (at_time 이하 데이터만 = 스포일러 방지)
  - predict.py  : 부스터 로드 + 예측 + isotonic 보정(표시확률)
  - models/     : 학습 산출물(final 4개 .txt + calibration + unity_contract)
"""
