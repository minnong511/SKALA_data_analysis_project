# 신약 후보물질 독성 예측

분자 구조와 물성으로 독성 여부를 예측합니다.

## 프로젝트

| 항목 | 내용 |
|---|---|
| 학습 데이터 | 8,349개 분자 |
| 원본 경로 | `Data/input_data/train.csv` |
| 라벨 | 0 = 독성, 1 = 비독성 |
| 입력 피처 | ECFP, FCFP, PTFP, 물성, 총 3,091개 |
| 모델 | LightGBM, XGBoost, ExtraTrees, 로지스틱 회귀, 앙상블 |
| 검증 | 안쪽, 바깥 Scaffold 5-Fold |
| 현재 목표 | 비독성 F1 ≥ 0.82에서 독성 Recall 최대화 |
| 희망 F1 | 0.83 이상 |

## 실험 결과

바깥 5-Fold 평균입니다. 기존 전체 데이터 탐색 이력이 있는 내부 검증 결과입니다.

| 실험 | 비독성 F1 | 독성 Recall |
|---|---:|---:|
| F1 우선 기준선 | 0.7974 | 0.6132 |
| Recall 0.90 기준선 | 0.6639 | 0.9062 |
| Optuna, Recall 0.87 조건 | 0.7202 | 0.8684 |
| 피처 선택 없는 대조군 | 0.7174 | 0.8708 |
| 피처 선택 포함 탐색 | 0.7182 | 0.8708 |

피처 선택의 F1 개선은 약 0.0008입니다. 새 목표인 F1 ≥ 0.82 탐색은 아직 실행하지 않았습니다.

## 파일

| 파일 | 용도 |
|---|---|
| [EDA.ipynb](EDA.ipynb) | 데이터 탐색 |
| [Train.ipynb](Train.ipynb) | 앙상블 기준선 |
| [Search.ipynb](Search.ipynb) | 랜덤 탐색 |
| [Optuna.ipynb](Optuna.ipynb) | Optuna TPE 탐색 |
| [FeatureSelection.ipynb](FeatureSelection.ipynb) | ANOVA, gain 기반 피처 선택 비교 |
| [analysis/f1_floor_search.py](analysis/f1_floor_search.py) | F1 하한 조건의 Recall 최적화 |

## 실행

```bash
python -m pip install -r requirements-training.txt

# 기존 Optuna 결과가 없을 때 먼저 실행
python -u -m analysis.optuna_recall --trials 20

# 새 목표의 계획 확인, 추가 탐색
python -m analysis.f1_floor_search --dry-run
python -u -m analysis.f1_floor_search --trials 12
```

새 탐색은 같은 Fold의 기존 안쪽 예측을 재활용하고, Fold당 12개 후보를 추가 평가합니다.

| 항목 | 안내 |
|---|---|
| 새 결과 경로 | `Data/training/f1floor82_recall_v1/` |
| 중단 후 재개 | 같은 명령에 `--resume` 추가 |
| 주요 출력 | Fold별 점수, 예측, 선택 기록, Optuna DB |
| 데이터, 학습 결과 | `Data/`에 보관, Git 미포함 |
| F1 하한 미달 | 최고 F1을 진단용으로 기록, 목표 미달 표시 |

전처리와 피처 선택은 각 학습 Fold 안에서만 수행합니다. 바깥 평가에서 목표 달성을 보장하지 않습니다.
