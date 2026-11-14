# Uni-D 데이터톤 배터리 열폭주 6단계 분류

이 문서는 DAKER 최종 선택 파일 `submission_final.csv`에 대응하는 제출 패키지 초안입니다. 현재 확보된 가중치는 13차 CatBoost, LightGBM 보완 모델 5개, 4·5·6단계 ExtraTrees 모델 5개입니다. 34차와 2·3단계 전용 모델의 가중치는 확보 및 재현 검증이 남아 있습니다.

## 실행 환경

| 항목 | 실제 사용 환경 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 5080 |
| 의존성 | `requirements.txt` |

```bash
python -m pip install -r requirements.txt
```

## 모델과 특징

- 13차 CatBoost는 열화상 이미지 관련 28개와 센서·시간 특징 21개를 포함한 총 49개 특징으로 1~6단계 확률을 예측합니다. 열화상 특징에는 2×2 구역별 RGB 평균, 밝은 영역의 중심과 퍼짐, 이미지 요약 통계가 포함됩니다.
- Stage23 B는 34차 6단계 모델과 2·3단계 전용 분류기를 사용합니다. 2·3단계 확률의 합을 유지하면서 두 단계 간 확률을 강도 0.75로 재배분한 후, 13차 CatBoost 확률과 50:50으로 결합합니다.
- Stage23 B 확률 90%와 LightGBM 보완 모델 확률 10%를 결합합니다. 4·5·6단계 확률의 합을 유지하면서 해당 세 단계의 조건부 확률에 ExtraTrees 전용 모델을 강도 0.75로 반영합니다. 마지막에 확률이 가장 높은 단계 하나를 선택합니다.
- 시계열 특징은 같은 실험의 현재·과거 관측과 초 단위 경과 시간을 사용합니다. 기본 특징은 과거 30초 통계를 사용합니다. 사건 이력 특징은 15·60·180·600초 구간의 통계, 첫 관측 대비 및 과거 극값 대비 변화, 이벤트 발생 이후 경과 시간을 포함합니다.
- 외부 사전학습 이미지 모델은 사용하지 않았습니다.

## 학습 및 검증

공식 Train으로 학습하고, 실험 ID 기준 5개 폴드로 분할하여 OOF 예측으로 조합을 비교했습니다. 공식 Validation은 별도 성능 확인에 사용했습니다. Test는 추론에 사용합니다. 데이터와 예측 확률 캐시는 팀 서버에 보관합니다.

## 가중치 배치

| 파일 | 용도 |
|---|---|
| `weights/model_rgb_2x2_bright_catboost.joblib` | 13차 6단계 CatBoost |
| `weights/leaves15_fold1.joblib` ~ `weights/leaves15_fold5.joblib` | 6단계 LightGBM 보완 모델 |
| `weights/expert456_extra_fold1.joblib` ~ `weights/expert456_extra_fold5.joblib` | 4·5·6단계 ExtraTrees 전용 모델 |
| 34차 5개 및 2·3단계 전용 5개 | 팀원이 전달한 실제 파일명과 배치 경로를 확인한 뒤 기재 |

웹 업로드 한도를 넘는 가중치가 있다면 다운로드 링크와 다운로드 후 둘 파일명을 이 절에 기재합니다.

## 추론 인터페이스

패키지 연결과 재현 검증이 완료된 뒤 아래 명령으로 실행합니다. `--input`은 제공 데이터의 추론 CSV 경로입니다. `--image_root`는 해당 이미지의 루트 폴더입니다. `--output`에는 새 CSV 저장 경로를 지정합니다.

```bash
python test.py --input /path/to/test.csv --image_root /path/to/images --output /path/to/submission.csv
```

출력은 입력 행 순서의 `sample_id,label` 두 열이며, 라벨은 1~6 중 하나입니다. 최종 배포 전에 저장된 가중치로 공식 Test 3,451행을 추론해 최종 선택 파일과 ID·라벨을 전부 대조합니다. 실험별 입력에서도 공통 행의 일치 여부를 점검합니다. 누락된 가중치를 연결한 뒤 학습 실행 명령과 검증 결과를 이 문서에 확정합니다.
