# Uni-D 배터리 열폭주 6단계 분류 — 베이스라인 v7 (제출 템플릿)

> 참가팀은 이 README를 자기 팀 내용으로 바꿔서 제출하십시오. `<>` 부분을 채우면 됩니다.

## 팀 정보

- 팀명: 6팀
- 최종 선택 제출 파일: submission_final.csv, Private 복원 대상

## 개발 환경

- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- GPU: NVIDIA GeForce RTX 5080
- 라이브러리: `requirements.txt` 참고

## 파일 구성

| 파일           | 역할                                                                       |
| -------------- | -------------------------------------------------------------------------- |
| `model.py`     | 피처 생성(이미지 통계·센서·과거 방향 시계열)과 모델 정의                   |
| `train.py`     | Train으로 학습, Validation으로 평가, 가중치 저장                           |
| `test.py`      | 입력 CSV에 들어 있는 행만으로 추론하고 제출 파일 생성                      |
| `model.joblib` | 학습된 가중치. 재학습 없이 `test.py`만으로 결과를 재현할 수 있어야 합니다. |

## 실행 방법

```bash
pip install -r requirements.txt

# 학습 (선택: --refit_train_val 을 주면 Train+Validation으로 최종 학습)
python train.py --data_dir /home/ubuntu/data/battery_data_light --model_out model.joblib

# 추론
python test.py --input /home/ubuntu/data/battery_data_light/test/test.csv \
               --image_root /home/ubuntu/data/battery_data_light/test \
               --model model.joblib --output submission.csv \
               --sample_submission /home/ubuntu/data/battery_data_light/sample_submission.csv
```

## 학습·검증 구성

- 학습 데이터: 공식 Train만 사용
- 검증 방법: Train에서 experiment_id 기준 GroupKFold(5)로 생성한 OOF 예측으로 조합을 비교하고, 공식 Validation에서 별도로 성능을 확인했습니다.
- 검증 점수: Validation Macro F1은 0.7675830222

## 시계열 피처 구성

- 같은 experiment_id 안에서 relative_time_s를 기준으로 현재·과거 관측을 사용했습니다. 기본 피처는 과거 30초 구간 통계와 관측값 차이를 사용합니다. 사건 이력 피처는 15·60·180·600초 구간 통계, 첫 관측 및 현재까지의 극값 대비 변화, 이벤트 발생 이후 경과 시간을 포함합니다.
- 미래 시점 관측과 실험 전체 기준 통계는 사용하지 않습니다.

## 사용한 사전학습 모델

- 없음
