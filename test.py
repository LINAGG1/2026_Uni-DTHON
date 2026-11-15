"""Predict requested CSV rows from saved Uni-D weights, without training."""

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

import model as M
from sensor_features_v3 import build_sensor_features_v3


class SensorCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


def make_sequences(frame, bundle):
    sensors = bundle["sensors"]
    seq_len = bundle["seq_len"]
    window = bundle["window_seconds"]

    values = (
        frame[sensors]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )
    values = np.clip(bundle["scaler"].transform(values), -10, 10)

    times = pd.to_numeric(
        frame["relative_time_s"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    result = np.zeros(
        (len(frame), seq_len, len(sensors) + 1),
        dtype=np.float32,
    )

    for indices in frame.groupby(
        "experiment_id", sort=False
    ).indices.values():
        indices = np.asarray(indices)
        idx = indices[np.argsort(times[indices], kind="stable")]
        t = times[idx]

        for j, row_idx in enumerate(idx):
            start = np.searchsorted(
                t, t[j] - window, side="left"
            )
            start = max(start, j - seq_len + 1)
            history = idx[start:j + 1]
            n = len(history)

            result[row_idx, -n:, :-1] = values[history]
            result[row_idx, -n:, -1] = 1.0

    return result


DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights"
EXPERT_STAGES = np.arange(4, 7)
PREDICTION_THREADS = 4
RGB_WEIGHT = "model_rgb_2x2_bright_catboost.joblib"


def validate_probabilities(values, n_rows, n_classes):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n_rows, n_classes):
        raise ValueError(f"예측 확률 크기가 다릅니다: {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("예측 확률에 음수 또는 비정상 값이 있습니다.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("예측 확률의 행별 합이 1이 아닙니다.")
    return values


def resolve_weights(weights, model):
    """Accept a training index, weight directory, or RGB bundle with fold files.

    train.py --model_out writes an index pointing to 21 companion weights.
    For the supplied pretrained package, a missing model.joblib still selects
    its sibling weights/ directory. Other missing paths are errors.
    """
    rgb_path = None
    if model is not None:
        if model.is_dir():
            if weights is not None and weights.resolve() != model.resolve():
                raise ValueError("--model과 --weights가 서로 다른 폴더입니다.")
            weights = model
        elif model.is_file():
            entry = joblib.load(model)
            if isinstance(entry, dict) and entry.get("format") == M.MODEL_INDEX_FORMAT:
                directory = entry.get("weights_dir")
                if not isinstance(directory, str) or not directory.strip():
                    raise ValueError("모델 인덱스에 weights_dir가 없습니다.")
                indexed_weights = model.resolve().parent / directory
                if weights is not None and weights.resolve() != indexed_weights.resolve():
                    raise ValueError("--model 인덱스와 --weights가 서로 다른 폴더입니다.")
                weights = indexed_weights
            else:
                rgb_path = model
                if weights is None:
                    weights = model.parent / "weights"
                    if not weights.is_dir():
                        weights = model.parent
        elif model.name == "model.joblib":
            weights = weights if weights is not None else model.parent / "weights"
            print(f"단일 model.joblib 대신 저장된 앙상블 가중치 사용: {weights}", file=sys.stderr)
        else:
            raise FileNotFoundError(f"모델 경로가 없습니다: {model}")
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    rgb_path = rgb_path if rgb_path is not None else weights / RGB_WEIGHT
    paths = [rgb_path]
    for fold in range(1, 6):
        paths.extend([
            weights / f"v34_fold{fold}.cbm",
            weights / f"stage23_fold{fold}.cbm",
            weights / f"leaves15_fold{fold}.joblib",
            weights / f"expert456_extra_fold{fold}.joblib",
        ])
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("필요한 가중치가 없습니다: " + ", ".join(missing))
    return weights, rgb_path


def predict_ordered(model, features, names, classes):
    order = np.argsort(model.classes_)
    if not np.array_equal(np.asarray(model.classes_)[order], classes):
        raise ValueError(f"모델의 클래스 순서를 확인하세요: {model.classes_}")
    X = features.loc[:, names]
    if isinstance(model, CatBoostClassifier):
        values = model.predict_proba(X, thread_count=PREDICTION_THREADS)
    else:
        values = model.predict_proba(X)
    return validate_probabilities(values[:, order], len(features), len(classes))


def catboost_folds(weights, stem, features, classes):
    for fold in range(1, 6):
        fitted = CatBoostClassifier()
        fitted.load_model(str(weights / f"{stem}_fold{fold}.cbm"))
        yield predict_ordered(fitted, features, fitted.feature_names_, classes)


def predict_from_features(frame, rgb_features, raw_v34_features, event_features, weights,
                          rgb_path=None):
    """Combine predictions from the five fitted branches."""
    bundle = joblib.load(rgb_path if rgb_path is not None else weights / RGB_WEIGHT)
    if not isinstance(bundle, dict) or not {"pipeline", "features"}.issubset(bundle):
        raise ValueError("RGB 모델은 pipeline과 features를 포함한 joblib이어야 합니다.")
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
    for fold in range(1, 6):
        fitted = joblib.load(weights / f"leaves15_fold{fold}.joblib")
        fitted.set_params(n_jobs=PREDICTION_THREADS)
        leaves += validate_probabilities(
            M.predict_lightgbm(fitted, leaves_features), len(frame), len(M.LABELS)
        ) / 5
        expert_features = M.make_extratrees_features(common_features, event_features, fitted)
        fitted = joblib.load(weights / f"expert456_extra_fold{fold}.joblib")
        # A fixed tree summation order prevents thread scheduling from changing
        # floating-point accumulation across repeated or split invocations.
        fitted.set_params(n_jobs=1)
        expert += predict_ordered(
            fitted, expert_features, fitted.feature_names_in_, EXPERT_STAGES
        ) / 5
    return validate_probabilities(
        M.combine_probabilities(v34, p3, rgb, leaves, expert), len(frame), len(M.LABELS)
    )


def validate_frame(frame):
    from feature_sources.event_history import SIGNALS
    from feature_sources.v34.sensor_features import SENSOR_COLUMNS, CONDITION_COLUMNS

    required = {
        "sample_id", "experiment_id", "relative_time_s", "image_path", "meas_a",
        *SIGNALS, *SENSOR_COLUMNS, *CONDITION_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("입력 CSV에 필요한 열이 없습니다: " + ", ".join(sorted(missing)))
    for name in ("sample_id", "experiment_id", "image_path"):
        if frame[name].isna().any() or frame[name].astype("string").str.strip().eq("").any():
            raise ValueError(f"{name}에 누락 또는 빈 값이 있습니다.")
    if not frame["sample_id"].is_unique:
        raise ValueError("sample_id에 중복이 있습니다.")
    frame["relative_time_s"] = pd.to_numeric(frame["relative_time_s"], errors="raise")
    if not np.isfinite(frame["relative_time_s"].to_numpy(dtype=float)).all():
        raise ValueError("relative_time_s는 유한한 숫자여야 합니다.")
    if frame.duplicated(["experiment_id", "relative_time_s"]).any():
        raise ValueError("같은 실험에 중복 시각이 있습니다.")


def validate_template(path, ids):
    """Check format and coverage only; never add rows or use template labels."""
    sample = pd.read_csv(path, dtype={"sample_id": "string"})
    if list(sample.columns) != ["sample_id", "label"]:
        raise ValueError("sample_submission 열은 sample_id,label이어야 합니다.")
    if (sample["sample_id"].isna().any()
            or sample["sample_id"].str.strip().eq("").any()
            or not sample["sample_id"].is_unique):
        raise ValueError("sample_submission의 sample_id에 누락, 빈 값 또는 중복이 있습니다.")
    if not ids.isin(sample["sample_id"]).all():
        raise ValueError("sample_submission에 입력 sample_id 일부가 없습니다.")


def resolve_requested_images(frame, image_root):
    """Resolve each row independently, avoiding batch-dependent root selection."""
    roots = [image_root.resolve()]
    if roots[0].parent != roots[0]:
        roots.append(roots[0].parent)
    resolved = []
    missing = []
    for value in frame["image_path"]:
        relative = Path(str(value).replace("\\", "/"))
        path = next((root / relative for root in roots if (root / relative).is_file()), None)
        resolved.append(path)
        if path is None:
            missing.append(relative)
    if missing:
        # Only filenames explicitly requested by this CSV may be selected.
        requested_names = {path.name for path in missing}
        candidates = {name: set() for name in requested_names}
        for root in roots:
            for path in root.rglob("*"):
                if path.name in candidates and path.is_file():
                    candidates[path.name].add(path.resolve())
        for i, path in enumerate(resolved):
            if path is not None:
                continue
            name = Path(str(frame.iloc[i]["image_path"]).replace("\\", "/")).name
            matches = candidates[name]
            if not matches:
                raise FileNotFoundError(f"입력 CSV의 이미지를 찾을 수 없습니다: {frame.iloc[i]['image_path']}")
            if len(matches) != 1:
                raise ValueError(f"이미지 파일명이 여러 경로에 있습니다. 정확한 image_path가 필요합니다: {name}")
            resolved[i] = next(iter(matches))
    frame = frame.copy()
    frame["image_path"] = [str(path.resolve()) for path in resolved]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="앙상블 가중치 폴더 (기본: test.py 옆 weights)")
    parser.add_argument("--model", type=Path,
                        help="학습 인덱스 joblib, 가중치 폴더 또는 RGB joblib")
    parser.add_argument("--sample_submission", type=Path, help="출력 형식/ID 확인용 CSV")
    args = parser.parse_args()
    protected = [args.input, args.sample_submission, args.model]
    if any(path is not None and path.resolve() == args.output.resolve() for path in protected):
        parser.error("출력 경로는 입력 CSV, sample_submission 및 모델 경로와 달라야 합니다.")
    frame = pd.read_csv(args.input, dtype={"sample_id": "string", "experiment_id": "string"})
    validate_frame(frame)
    ids = frame["sample_id"]
    if args.sample_submission is not None:
        validate_template(args.sample_submission, ids)
    weights, rgb_path = resolve_weights(args.weights, args.model)
    if args.output.resolve() in {path.resolve() for path in weights.iterdir()} | {rgb_path.resolve()}:
        parser.error("출력 경로로 가중치 파일을 덮어쓸 수 없습니다.")
    if len(frame):
        if not args.image_root.is_dir():
            raise NotADirectoryError(f"이미지 폴더가 없습니다: {args.image_root}")
        # model.py has no n_jobs argument. Control image extraction's joblib
        # backend here to avoid spawning one process per CPU on the evaluator.
        frame = resolve_requested_images(frame, args.image_root)
        with joblib.parallel_backend("threading"):
            rgb_features, raw_v34_features, event_features = M.build_submission_features(
                frame, args.image_root, raw_v34=True
            )
        probabilities = predict_from_features(
            frame, rgb_features, raw_v34_features, event_features, weights, rgb_path
        )
        labels = M.LABELS[probabilities.argmax(axis=1)]
    else:
        labels = np.empty(0, dtype=int)
    output = pd.DataFrame({"sample_id": ids, "label": labels}).astype({"label": int})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"저장 완료: {args.output} ({len(output)}행)")


if __name__ == "__main__":
    main()
