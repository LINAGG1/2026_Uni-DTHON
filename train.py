"""Train all five branches required by the Uni-D inference ensemble."""

import argparse
import os
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


def weight_filenames():
    names = [RGB_WEIGHT]
    for fold in range(1, 6):
        names.extend([
            f"v34_fold{fold}.cbm", f"stage23_fold{fold}.cbm",
            f"leaves15_fold{fold}.joblib", f"expert456_extra_fold{fold}.joblib",
        ])
    return names


def load_training_data(data_dir, image_root=None, refit_train_val=False):
    from feature_sources.v34.model import resolve_image_paths

    frames = []
    experiment_sets = []
    splits = ("train", "validation") if refit_train_val else ("train",)
    for split in splits:
        root = data_dir / split
        frame = pd.read_csv(root / f"{split}.csv",
                            dtype={"sample_id": "string", "experiment_id": "string"})
        if frame["experiment_id"].isna().any():
            raise ValueError(f"{split} experiment_id에 누락값이 있습니다.")
        experiments = set(frame["experiment_id"])
        if any(experiments & previous for previous in experiment_sets):
            raise ValueError("Train과 Validation에 같은 experiment_id가 있습니다.")
        experiment_sets.append(experiments)
        # Resolve before concatenation so both splits' images use their own roots.
        roots = ([image_root] if image_root is not None else []) + [root, data_dir]
        frame["image_path"] = resolve_image_paths(frame, roots).map(lambda p: str(p.resolve()))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if frame["sample_id"].isna().any() or not frame["sample_id"].is_unique:
        raise ValueError("학습 sample_id에 누락 또는 중복이 있습니다.")
    return frame


def training_categories(common):
    return {
        name: sorted(common[name].replace(-999, np.nan).dropna().unique())
        for name in common.columns
        if name in M.CATEGORICAL_COLUMNS
    }


def train_branches(frame, image_root, weights):
    labels = pd.to_numeric(frame["label"], errors="raise")
    if set(labels) != set(M.LABELS):
        raise ValueError("Train에 정수 라벨 1~6이 모두 필요합니다.")
    groups = frame["experiment_id"]
    if groups.isna().any() or groups.nunique() < 5:
        raise ValueError("GroupKFold에 누락 없는 실험 ID가 5개 이상 필요합니다.")

    y = labels.to_numpy(dtype=int)
    folds = list(GroupKFold(n_splits=5).split(frame, y, groups))
    for fold, (fit, _) in enumerate(folds, start=1):
        if set(y[fit]) != set(M.LABELS):
            raise ValueError(f"Fold {fold}의 학습 행에 라벨 1~6이 모두 필요합니다.")

    # v34 was trained on raw NaN; the other shared branches require -999.
    with joblib.parallel_backend("threading"):
        rgb, raw_common, events = M.build_submission_features(frame, image_root, raw_v34=True)
    common = M.prepare_common_features(raw_common)
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

    for fold, (fit, _) in enumerate(folds, start=1):
        v34 = CatBoostClassifier(
            loss_function="MultiClass", iterations=1200, depth=8, learning_rate=0.05,
            class_weights=[1, 1, 1, 1.2, 1.5, 1], random_seed=42,
            thread_count=4, verbose=False, allow_writing_files=False,
        ).fit(raw_common.iloc[fit], y[fit])
        if not np.array_equal(v34.classes_, M.LABELS):
            raise ValueError(f"Fold {fold}의 v34 클래스 순서를 확인하세요.")
        v34.save_model(str(weights / f"v34_fold{fold}.cbm"))

        stage23_rows = fit[np.isin(y[fit], (2, 3))]
        stage23 = CatBoostClassifier(
            loss_function="Logloss", iterations=1200, depth=8, learning_rate=0.05,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
        ).fit(common.iloc[stage23_rows], (y[stage23_rows] == 3).astype(int))
        if not np.array_equal(stage23.classes_, (0, 1)):
            raise ValueError(f"Fold {fold}의 Stage23 클래스 순서를 확인하세요.")
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
        print(f"Fold {fold}/5 v34, Stage23, LightGBM 및 ExtraTrees 저장 완료", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True, type=Path)
    parser.add_argument("--image_root", "--image-root", dest="image_root", type=Path)
    parser.add_argument("--weights", type=Path, help="새 가중치 폴더 (기존 가중치는 덮어쓰지 않음)")
    parser.add_argument("--model_out", "--model-out", dest="model_out", type=Path,
                        help="test.py --model에 전달할 앙상블 인덱스 joblib")
    parser.add_argument("--refit_train_val", "--refit-train-val", dest="refit_train_val",
                        action="store_true", help="Train+Validation으로 최종 학습")
    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        weights = (args.model_out.parent / f"{args.model_out.stem}_weights"
                   if args.model_out is not None else DEFAULT_WEIGHTS)
    if args.model_out is not None:
        if args.model_out.exists():
            raise FileExistsError(f"기존 모델 인덱스는 덮어쓰지 않습니다: {args.model_out}")
        if args.model_out.resolve() in {(weights / name).resolve() for name in weight_filenames()}:
            raise ValueError("--model_out은 개별 가중치 파일과 다른 경로여야 합니다.")
    existing = [name for name in weight_filenames() if (weights / name).exists()]
    if existing:
        raise FileExistsError(f"다른 --weights 폴더를 지정하세요. 기존 가중치: {', '.join(existing)}")

    frame = load_training_data(args.data_dir, args.image_root, args.refit_train_val)
    weights.mkdir(parents=True, exist_ok=True)
    train_branches(frame, args.image_root or args.data_dir, weights)
    missing = [name for name in weight_filenames() if not (weights / name).is_file()]
    if missing:
        raise RuntimeError("학습 가중치 생성 누락: " + ", ".join(missing))
    if args.model_out is not None:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "format": M.MODEL_INDEX_FORMAT,
            "weights_dir": os.path.relpath(weights.resolve(), args.model_out.parent.resolve()),
            "training_splits": ["train", "validation"] if args.refit_train_val else ["train"],
            "training_rows": len(frame), "training_experiments": frame["experiment_id"].nunique(),
        }, args.model_out)
        print(f"모델 인덱스 저장 완료: {args.model_out}", flush=True)
    print(f"가중치 21개 저장 완료: {weights}", flush=True)


if __name__ == "__main__":
    main()
