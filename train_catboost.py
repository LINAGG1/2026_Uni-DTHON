
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold

from train import load_split, macro_f1

DATA_DIR = Path.home() / "data/battery_data_light"

print("데이터 로딩 중...", flush=True)
train, X_tr, y_tr = load_split(DATA_DIR, "train")
val, X_va, y_va = load_split(DATA_DIR, "validation")

assert set(train.experiment_id).isdisjoint(val.experiment_id)

params = {
    "iterations": 700,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "MultiClass",
    "verbose": False,
    "random_seed": 42,
    "thread_count": 4,
}

scores = []
n_splits = min(5, train.experiment_id.nunique())

print("GroupKFold 검증 시작", flush=True)

for fold, (tr_idx, va_idx) in enumerate(
    GroupKFold(n_splits=n_splits).split(
        X_tr, y_tr, groups=train.experiment_id
    ), 1
):
    model = CatBoostClassifier(**params)
    model.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
    pred = model.predict(X_tr.iloc[va_idx]).ravel().astype(int)

    score = macro_f1(y_tr.iloc[va_idx], pred)
    scores.append(score)
    print(f"Fold {fold}: {score:.4f}", flush=True)

print(f"GroupKFold Macro F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("전체 Train 학습 중...", flush=True)
model = CatBoostClassifier(**params)
model.fit(X_tr, y_tr)

pred = model.predict(X_va).ravel().astype(int)
score = macro_f1(y_va, pred)

print(f"Validation Macro F1: {score:.4f}", flush=True)

joblib.dump(
    {
        "pipeline": model,
        "features": list(X_tr.columns),
        "class_weight": None,
    },
    "model_catboost.joblib",
)

print("모델 저장: model_catboost.joblib", flush=True)
