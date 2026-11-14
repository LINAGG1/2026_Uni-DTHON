"""Train five ensemble branches with experiment-disjoint OOF and Validation reports."""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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


def load_labeled_split(data_dir, split, image_root=None):
    from feature_sources.v34.model import resolve_image_paths

    if split not in ("train", "validation"):
        raise ValueError("학습·평가에는 Train 또는 Validation만 사용할 수 있습니다.")
    root = data_dir / split
    frame = pd.read_csv(root / f"{split}.csv",
                        dtype={"sample_id": "string", "experiment_id": "string"})
    if frame.empty:
        raise ValueError(f"{split} 데이터가 비어 있습니다.")
    for name in ("sample_id", "experiment_id"):
        if frame[name].isna().any() or frame[name].str.strip().eq("").any():
            raise ValueError(f"{split} {name}에 누락 또는 빈 값이 있습니다.")
    if not frame["sample_id"].is_unique:
        raise ValueError(f"{split} sample_id에 중복이 있습니다.")
    labels = pd.to_numeric(frame["label"], errors="raise")
    if not labels.isin(M.LABELS).all():
        raise ValueError(f"{split}의 라벨은 정수 1~6이어야 합니다.")
    frame["label"] = labels.astype(int)
    roots = ([image_root] if image_root is not None else []) + [root, data_dir]
    frame["image_path"] = resolve_image_paths(frame, roots).map(lambda p: str(p.resolve()))
    return frame


def ensure_disjoint(train, validation):
    for name in ("sample_id", "experiment_id"):
        if set(train[name]) & set(validation[name]):
            raise ValueError(f"Train과 Validation에 같은 {name}가 있습니다.")


def load_training_data(data_dir, image_root=None, refit_train_val=False):
    train = load_labeled_split(data_dir, "train", image_root)
    if not refit_train_val:
        return train.reset_index(drop=True)
    validation = load_labeled_split(data_dir, "validation", image_root)
    ensure_disjoint(train, validation)
    return pd.concat([train, validation], ignore_index=True)


def training_categories(common):
    return {
        name: sorted(common[name].replace(-999, np.nan).dropna().unique().tolist())
        for name in common.columns if name in M.CATEGORICAL_COLUMNS
    }


def encode_categories(features, categories):
    encoded = features.copy()
    for name, levels in categories.items():
        encoded[name] = pd.Categorical(encoded[name], categories=levels)
    return encoded


def rgb_estimator():
    return CatBoostClassifier(
        loss_function="MultiClass", iterations=400, depth=5, learning_rate=0.05,
        random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
    )


def build_features(frame, image_root):
    with joblib.parallel_backend("threading"):
        rgb, raw, events = M.build_submission_features(frame, image_root, raw_v34=True)
    common = M.prepare_common_features(raw)
    leaves = M.make_lightgbm_features(frame, common)
    if leaves.shape[1] != 203 or common.shape[1] + events.shape[1] != 797:
        raise ValueError("LightGBM / ExtraTrees의 특징 수가 예상과 다릅니다.")
    return {"rgb": rgb, "raw": raw, "common": common, "events": events, "leaves": leaves}


def ordered_probabilities(estimator, features, classes):
    classes = np.asarray(classes)
    order = np.argsort(estimator.classes_)
    if not np.array_equal(np.asarray(estimator.classes_)[order], classes):
        raise ValueError(f"모델 클래스가 예상과 다릅니다: {estimator.classes_}")
    if isinstance(estimator, CatBoostClassifier):
        probabilities = estimator.predict_proba(features, thread_count=4)[:, order]
    else:
        probabilities = estimator.predict_proba(features)[:, order]
    if (probabilities.shape != (len(features), len(classes))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("예측 확률이 올바르지 않습니다.")
    return probabilities


def scores(actual, predicted, labels=M.LABELS):
    labels = np.asarray(labels)
    return {
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "class_f1": dict(zip(map(str, labels), f1_score(actual, predicted, labels=labels,
                                                       average=None, zero_division=0).tolist())),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "label_order": labels.tolist(), "rows": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
    }


def blank_probabilities(rows):
    return {"rgb": np.full((rows, 6), np.nan), "v34": np.full((rows, 6), np.nan),
            "stage23": np.full(rows, np.nan), "leaves": np.full((rows, 6), np.nan),
            "expert": np.full((rows, 3), np.nan)}


def combine(branches):
    return M.combine_probabilities(branches["v34"], branches["stage23"], branches["rgb"],
                                   branches["leaves"], branches["expert"])


def branch_scores(actual, branches):
    result = {name: scores(actual, M.LABELS[branches[name].argmax(axis=1)])
              for name in ("rgb", "v34", "leaves")}
    for name, labels in (("stage23", (2, 3)), ("expert", EXPERT_STAGES)):
        mask = np.isin(actual, labels)
        if not mask.any():
            result[name] = {"status": "no_relevant_labels"}
            continue
        predicted = (np.where(branches[name][mask] >= .5, 3, 2) if name == "stage23"
                     else np.asarray(labels)[branches[name][mask].argmax(axis=1)])
        result[name] = scores(actual[mask], predicted, labels)
        result[name]["scope"] = "Diagnostic only: actual labels in this expert's classes"
    return result


def prediction_table(frame, branches, fold_ids=None):
    probabilities = combine(branches)
    table = frame[["sample_id", "experiment_id", "label"]].copy()
    if fold_ids is not None:
        table["fold"] = fold_ids
    table["predicted_label"] = M.LABELS[probabilities.argmax(axis=1)]
    table[[f"prob_{label}" for label in M.LABELS]] = probabilities
    return table, probabilities


def train_branches(frame, image_root, weights, validation=None, evaluation_dir=None,
                   refit_train_val=False):
    y = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=int)
    if set(y) != set(M.LABELS):
        raise ValueError("학습 데이터에 정수 라벨 1~6이 모두 필요합니다.")
    groups = frame["experiment_id"]
    if groups.isna().any() or groups.nunique() < 5:
        raise ValueError("GroupKFold에는 실험 ID가 5개 이상 필요합니다.")
    folds = list(GroupKFold(n_splits=5).split(frame, y, groups))
    for fold, (fit, _) in enumerate(folds, 1):
        if set(y[fit]) != set(M.LABELS):
            raise ValueError(f"Fold {fold}의 학습 행에 라벨 1~6이 모두 필요합니다.")
    if validation is not None:
        if refit_train_val:
            raise ValueError("학습에 포함한 Validation을 독립 평가에 사용할 수 없습니다.")
        ensure_disjoint(frame, validation)
    features = build_features(frame, image_root)
    val_features = build_features(validation, image_root) if validation is not None else None
    oof = blank_probabilities(len(frame))
    val = blank_probabilities(len(validation)) if validation is not None else None
    fold_ids = np.zeros(len(frame), dtype=int)
    records = []

    # The deployed RGB branch uses all training data. It never supplies OOF predictions.
    rgb_model = rgb_estimator().fit(features["rgb"], y)
    joblib.dump({"pipeline": rgb_model, "features": list(features["rgb"].columns), "class_weight": None},
                weights / RGB_WEIGHT)
    if val is not None:
        val["rgb"] = ordered_probabilities(rgb_model, val_features["rgb"], M.LABELS)
        for name in ("v34", "stage23", "leaves", "expert"):
            val[name].fill(0)
    print("전체 학습 데이터의 RGB 가중치 저장 완료", flush=True)

    for fold, (fit, held) in enumerate(folds, 1):
        print(f"Fold {fold}/5 학습 및 OOF 평가 시작", flush=True)
        categories = training_categories(features["common"].iloc[fit])
        rgb_fold = rgb_estimator().fit(features["rgb"].iloc[fit], y[fit])
        oof["rgb"][held] = ordered_probabilities(rgb_fold, features["rgb"].iloc[held], M.LABELS)

        v34 = CatBoostClassifier(
            loss_function="MultiClass", iterations=1200, depth=8, learning_rate=.05,
            class_weights=[1, 1, 1, 1.2, 1.5, 1], random_seed=42, thread_count=4,
            verbose=False, allow_writing_files=False,
        ).fit(features["raw"].iloc[fit], y[fit])
        v34.save_model(str(weights / f"v34_fold{fold}.cbm"))
        stage_rows = fit[np.isin(y[fit], (2, 3))]
        stage23 = CatBoostClassifier(
            loss_function="Logloss", iterations=1200, depth=8, learning_rate=.05,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
        ).fit(features["common"].iloc[stage_rows], (y[stage_rows] == 3).astype(int))
        stage23.save_model(str(weights / f"stage23_fold{fold}.cbm"))
        leaves = LGBMClassifier(
            n_estimators=700, learning_rate=.035, num_leaves=15, min_child_samples=60,
            reg_lambda=5., colsample_bytree=.85, subsample=.85, subsample_freq=1,
            n_jobs=4, verbosity=-1, random_state=42, deterministic=True, force_col_wise=True,
            class_weight={1: 1, 2: 1, 3: 1, 4: 1.2, 5: 1.5, 6: 1},
        ).fit(encode_categories(features["leaves"].iloc[fit], categories), y[fit],
              categorical_feature=list(categories))
        joblib.dump(leaves, weights / f"leaves15_fold{fold}.joblib")
        expert_fit = M.make_extratrees_features(features["common"].iloc[fit], features["events"].iloc[fit], leaves)
        active = np.isin(y[fit], EXPERT_STAGES)
        expert = ExtraTreesClassifier(
            n_estimators=500, max_features=.65, min_samples_leaf=4, n_jobs=4, random_state=2026,
        ).fit(expert_fit.iloc[np.flatnonzero(active)], y[fit][active])
        expert.set_params(n_jobs=1)
        if not np.array_equal(expert.classes_, EXPERT_STAGES):
            raise ValueError(f"Fold {fold}의 ExtraTrees 클래스가 예상과 다릅니다.")
        joblib.dump(expert, weights / f"expert456_extra_fold{fold}.joblib")

        for subset, target, idx in ((features, oof, held), (val_features, val, None)):
            if subset is None:
                continue
            part = {k: (v.iloc[idx] if idx is not None else v) for k, v in subset.items()}
            batch = {
                "v34": ordered_probabilities(v34, part["raw"], M.LABELS),
                "stage23": ordered_probabilities(stage23, part["common"], (0, 1))[:, 1],
                "leaves": M.predict_lightgbm(leaves, part["leaves"]),
                "expert": ordered_probabilities(expert, M.make_extratrees_features(
                    part["common"], part["events"], leaves), EXPERT_STAGES),
            }
            for name, probability in batch.items():
                if idx is None:
                    target[name] += probability / 5
                else:
                    target[name][idx] = probability
        fold_ids[held] = fold
        fold_branches = {name: probability[held] for name, probability in oof.items()}
        result = scores(y[held], M.LABELS[combine(fold_branches).argmax(axis=1)])
        records.append({"fold": fold, "fit_rows": len(fit), "held_out_rows": len(held),
                        "fit_experiments": sorted(groups.iloc[fit].unique().tolist()),
                        "held_out_experiments": sorted(groups.iloc[held].unique().tolist()),
                        "categories": categories, "ensemble": result})
        print(f"Fold {fold}/5 OOF Macro F1: {result['macro_f1']:.6f}", flush=True)

    evaluation_dir = evaluation_dir or weights / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    table, probabilities = prediction_table(frame, oof, fold_ids)
    table.to_csv(evaluation_dir / "oof_predictions.csv", index=False)
    np.savez_compressed(evaluation_dir / "oof_branch_probabilities.npz", **oof,
                        sample_id=frame["sample_id"].to_numpy(str), fold=fold_ids, label=y)
    report = {
        "training_splits": ["train", "validation"] if refit_train_val else ["train"],
        "preprocessing": "Fold category vocabularies fitted only on fit experiments; fixed -999 or native NaN handling; no Test data read",
        "oof_scope": "train_validation" if refit_train_val else "train",
        "oof": scores(y, M.LABELS[probabilities.argmax(axis=1)]),
        "oof_branches": branch_scores(y, oof), "folds": records,
        "rgb_oof": "Separate RGB model trained on each fit fold; full-training RGB used only for deployment and independent Validation",
        "validation": None,
        "validation_status": "included_in_training_not_independent" if refit_train_val else "not_available",
        "ensemble_parameters": {"stage23_strength": .75, "rgb_weight": .5,
                                "leaves_weight": .1, "expert456_strength": .75},
        "note": "Fixed ensemble configuration, not nested hyperparameter selection. Re-training metrics need not equal historical saved-weight scores.",
    }
    if val is not None:
        val_y = validation["label"].to_numpy(int)
        val_table, val_probabilities = prediction_table(validation, val)
        val_table.to_csv(evaluation_dir / "validation_predictions.csv", index=False)
        report["validation"] = scores(val_y, M.LABELS[val_probabilities.argmax(axis=1)])
        report["validation_branches"] = branch_scores(val_y, val)
        report["validation_status"] = "independent"
        print(f"Validation Macro F1: {report['validation']['macro_f1']:.6f}", flush=True)
    (evaluation_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"전체 OOF Macro F1: {report['oof']['macro_f1']:.6f}", flush=True)
    print(f"평가 기록 저장 완료: {evaluation_dir}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True, type=Path)
    parser.add_argument("--image_root", "--image-root", dest="image_root", type=Path)
    parser.add_argument("--weights", type=Path, help="새 가중치 폴더 (기존 가중치는 덮어쓰지 않음)")
    parser.add_argument("--model_out", "--model-out", dest="model_out", type=Path,
                        help="test.py --model에 전달할 앙상블 인덱스 joblib")
    parser.add_argument("--refit_train_val", "--refit-train-val", dest="refit_train_val", action="store_true",
                        help="Train+Validation 학습 및 해당 합집합의 OOF 평가; 독립 Validation 평가는 생략")
    args = parser.parse_args()
    weights = args.weights
    if weights is None:
        weights = args.model_out.parent / f"{args.model_out.stem}_weights" if args.model_out else DEFAULT_WEIGHTS
    if args.model_out is not None:
        if args.model_out.exists():
            raise FileExistsError(f"기존 모델 인덱스는 덮어쓰지 않습니다: {args.model_out}")
        if args.model_out.resolve() in {(weights / name).resolve() for name in weight_filenames()}:
            raise ValueError("--model_out은 개별 가중치 파일과 다른 경로여야 합니다.")
    existing = [name for name in weight_filenames() if (weights / name).exists()]
    if existing:
        raise FileExistsError(f"다른 --weights 폴더를 지정하세요. 기존 가중치: {', '.join(existing)}")
    frame = load_training_data(args.data_dir, args.image_root, args.refit_train_val)
    validation = None
    if not args.refit_train_val and (args.data_dir / "validation/validation.csv").is_file():
        validation = load_labeled_split(args.data_dir, "validation", args.image_root)
        ensure_disjoint(frame, validation)
    weights.mkdir(parents=True, exist_ok=True)
    train_branches(frame, args.image_root or args.data_dir, weights, validation=validation,
                   refit_train_val=args.refit_train_val)
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
