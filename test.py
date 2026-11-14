"""Run inference with the weights for the selected Uni-D submission."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import model as M


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
EXPERT_STAGES = np.arange(4, 7)


def predict_ordered(model, features, names, classes):
    order = np.argsort(model.classes_)
    if not np.array_equal(np.asarray(model.classes_)[order], classes):
        raise ValueError(f"모델의 클래스 순서를 확인하세요: {model.classes_}")
    return model.predict_proba(features.loc[:, names])[:, order]


def catboost_folds(weights, stem, features, classes):
    for fold in range(1, 6):
        fitted = CatBoostClassifier()
        fitted.load_model(str(weights / f"{stem}_fold{fold}.cbm"))
        yield predict_ordered(fitted, features, fitted.feature_names_, classes)


def predict_from_features(frame, rgb_features, raw_v34_features, event_features, weights):
    """Combine predictions from the five fitted branches."""
    bundle = joblib.load(weights / "model_rgb_2x2_bright_catboost.joblib")
    rgb = predict_ordered(bundle["pipeline"], rgb_features, bundle["features"], M.LABELS)
    v34 = np.zeros((len(frame), len(M.LABELS)))
    for probabilities in catboost_folds(weights, "v34", raw_v34_features, M.LABELS):
        v34 += probabilities / 5
    common_features = M.prepare_common_features(raw_v34_features)
    p3 = np.zeros(len(frame))
    for probabilities in catboost_folds(weights, "stage23", common_features, (0, 1)):
        p3 += probabilities[:, 1] / 5

    leaves_features = M.make_lightgbm_features(frame, common_features)
    leaves = np.zeros((len(frame), 6))
    expert = np.zeros((len(frame), 3))
    expert_features = None
    for fold in range(1, 6):
        fitted = joblib.load(weights / f"leaves15_fold{fold}.joblib")
        leaves += M.predict_lightgbm(fitted, leaves_features) / 5
        if expert_features is None:
            # The audit found the original category levels match Train-only levels.
            expert_features = M.make_extratrees_features(common_features, event_features, fitted)
        fitted = joblib.load(weights / f"expert456_extra_fold{fold}.joblib")
        expert += predict_ordered(
            fitted, expert_features, fitted.feature_names_in_, EXPERT_STAGES
        ) / 5
    return M.combine_probabilities(v34, p3, rgb, leaves, expert)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={"sample_id": "string", "experiment_id": "string"})
    ids = frame["sample_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError("sample_id에 누락, 빈 값 또는 중복이 있습니다.")
    if frame.empty:
        raise ValueError("추론할 입력 행이 없습니다.")
    build_features = getattr(M, "build_submission_features", None)
    if build_features is None:
        raise RuntimeError("model.py에 원본 CSV·이미지 특징 생성 함수를 연결해야 합니다.")
    rgb_features, raw_v34_features, event_features = build_features(
        frame, args.image_root, raw_v34=True
    )
    probabilities = predict_from_features(
        frame, rgb_features, raw_v34_features, event_features, args.weights
    )
    output = pd.DataFrame({"sample_id": ids, "label": M.LABELS[probabilities.argmax(axis=1)]})
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
