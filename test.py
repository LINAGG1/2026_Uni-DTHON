"""Train and save the five model branches used by test.py."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

import model as M


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
RGB_WEIGHT = "model_rgb_2x2_bright_catboost.joblib"
EXPERT_STAGES = (4, 5, 6)


def training_categories(common):
    return {
        name: sorted(common[name].replace(-999, np.nan).dropna().unique())
        for name in common.columns
        if name in M.CATEGORICAL_COLUMNS
    }


def train_branches(frame, image_root, weights, devices):
    labels = pd.to_numeric(frame["label"], errors="raise")
    if set(labels) != set(M.LABELS):
        raise ValueError("Train에 정수 라벨 1~6이 모두 필요합니다.")
    groups = frame["experiment_id"]
    if groups.isna().any() or groups.nunique() < 5:
        raise ValueError("GroupKFold에 누락 없는 실험 ID가 5개 이상 필요합니다.")

    rgb, raw_v34, events = M.build_submission_features(frame, image_root, raw_v34=True)
    common = M.prepare_common_features(raw_v34)
    leaves_X = M.make_lightgbm_features(frame, common)
    categories = training_categories(common)
    for name, levels in categories.items():
        leaves_X[name] = pd.Categorical(leaves_X[name], categories=levels)
    # Use the same column order and missing-value preprocessing as the experts.
    expert_X = pd.concat([
        common.reset_index(drop=True).replace(-999, np.nan),
        events.reset_index(drop=True).add_prefix("event_"),
    ], axis=1)
    for name, levels in categories.items():
        expert_X[name] = pd.Categorical(expert_X[name], categories=levels).codes.astype(float)
    expert_X = expert_X.fillna(-999).astype(np.float32)
    if leaves_X.shape[1] != 203 or expert_X.shape[1] != 797:
        raise ValueError(f"특징 수 확인 필요: LightGBM {leaves_X.shape[1]}, ExtraTrees {expert_X.shape[1]}")

    rgb_model = CatBoostClassifier(
        loss_function="MultiClass", iterations=400, depth=5, learning_rate=0.05,
        random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
    ).fit(rgb, labels)
    joblib.dump({"pipeline": rgb_model, "features": list(rgb.columns), "class_weight": None},
                weights / RGB_WEIGHT)
    print("13차 CatBoost 저장 완료", flush=True)

    y = labels.to_numpy(dtype=int)
    folds = GroupKFold(n_splits=5).split(leaves_X, y, groups)
    gpu_params = {
        "iterations": 1200,
        "depth": 8,
        "learning_rate": 0.05,
        "random_seed": 42,
        "task_type": "GPU",
        "devices": devices,
        "allow_writing_files": False,
    }
    for fold, (fit, _) in enumerate(folds, start=1):
        v34 = CatBoostClassifier(
            **gpu_params, loss_function="MultiClass",
            class_weights=[1, 1, 1, 1.2, 1.5, 1], verbose=200,
        ).fit(raw_v34.iloc[fit], y[fit])
        v34.save_model(str(weights / f"v34_fold{fold}.cbm"))

        rows23 = fit[np.isin(y[fit], (2, 3))]
        stage23 = CatBoostClassifier(**gpu_params, loss_function="Logloss", verbose=False)
        stage23.fit(common.iloc[rows23], (y[rows23] == 3).astype(int))
        stage23.save_model(str(weights / f"stage23_fold{fold}.cbm"))

        leaves = LGBMClassifier(
            n_estimators=700, learning_rate=0.035, num_leaves=15, min_child_samples=60,
            reg_lambda=5.0, colsample_bytree=0.85, subsample=0.85, subsample_freq=1,
            n_jobs=4, verbosity=-1, random_state=42, deterministic=True, force_col_wise=True,
            class_weight={1: 1, 2: 1, 3: 1, 4: 1.2, 5: 1.5, 6: 1},
        ).fit(leaves_X.iloc[fit], y[fit], categorical_feature=list(categories))
        joblib.dump(leaves, weights / f"leaves15_fold{fold}.joblib")

        rows = fit[np.isin(y[fit], EXPERT_STAGES)]
        expert = ExtraTreesClassifier(
            n_estimators=500, max_features=0.65, min_samples_leaf=4,
            n_jobs=4, random_state=2026,
        ).fit(expert_X.iloc[rows], y[rows])
        if not np.array_equal(expert.classes_, EXPERT_STAGES):
            raise ValueError(f"Fold {fold}의 ExtraTrees 학습 라벨을 확인하세요.")
        joblib.dump(expert, weights / f"expert456_extra_fold{fold}.joblib")
        print(f"Fold {fold}/5 | 34차, Stage23, LightGBM, ExtraTrees 저장 완료", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--devices", default="0", help="CatBoost GPU device selection (default: 0)")
    args = parser.parse_args()

    targets = [RGB_WEIGHT]
    targets.extend(f"{stem}_fold{fold}.joblib"
                   for stem in ("leaves15", "expert456_extra") for fold in range(1, 6))
    targets.extend(f"{stem}_fold{fold}.cbm"
                   for stem in ("v34", "stage23") for fold in range(1, 6))
    existing = [name for name in targets if (args.weights / name).exists()]
    if existing:
        raise FileExistsError(f"다른 --weights 폴더를 지정하세요. 기존 가중치: {', '.join(existing)}")

    train_dir = args.data_dir / "train"
    frame = pd.read_csv(train_dir / "train.csv", dtype={"sample_id": "string", "experiment_id": "string"})
    args.weights.mkdir(parents=True, exist_ok=True)
    train_branches(frame, args.image_root or train_dir, args.weights, args.devices)


if __name__ == "__main__":
    main()
