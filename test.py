import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "weights/model_sensor_v7_gpu.joblib",
    )
    parser.add_argument(
        "--cnn_model",
        type=Path,
        default=Path(__file__).resolve().parent
        / "weights/model_sensor_v16_cnn.joblib",
    )
    args = parser.parse_args()

    frame = pd.read_csv(
        args.input,
        dtype={"sample_id": "string", "experiment_id": "string"},
    )

    assert frame["sample_id"].is_unique

    # CatBoost: 이미지 + 센서 특징
    paths = M.resolve_image_paths(
        frame, [args.image_root, args.image_root.parent]
    )
    img = M.extract_image_features(frame, paths)
    X_base = M.build_features(frame, img)

    X_sensor = build_sensor_features_v3(frame)
    X_sensor = X_sensor.drop(
        columns=X_base.columns.intersection(X_sensor.columns),
        errors="ignore",
    )
    X = pd.concat(
        [X_base.reset_index(drop=True),
         X_sensor.reset_index(drop=True)],
        axis=1,
    )

    cat_bundle = joblib.load(args.model)
    cat = cat_bundle["pipeline"]

    raw = cat.predict_proba(X[cat_bundle["features"]])
    classes = np.asarray(cat.classes_).astype(int)
    cat_proba = np.column_stack([
        raw[:, np.where(classes == c)[0][0]]
        for c in range(1, 7)
    ])

    # CNN: 과거 센서 시퀀스
    cnn_bundle = joblib.load(args.cnn_model)
    seq = make_sequences(frame, cnn_bundle)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cnn = SensorCNN(len(cnn_bundle["sensors"]) + 1).to(device)
    cnn.load_state_dict(cnn_bundle["state_dict"])
    cnn.eval()

    cnn_probs = []
    with torch.no_grad():
        for start in range(0, len(seq), 256):
            xb = torch.from_numpy(
                seq[start:start + 256]
            ).to(device)
            probs = torch.softmax(cnn(xb), dim=1)
            cnn_probs.append(probs.cpu().numpy())

    cnn_proba = np.concatenate(cnn_probs)

    weight = cnn_bundle["best_cat_weight"]
    ensemble = weight * cat_proba + (1 - weight) * cnn_proba

    # 20차: 확률 보정 (한 번만 적용)
    ensemble[:, 3] *= 0.8
    ensemble[:, 4] *= 1.1

    # 최종 확률 저장
    np.savez(
        "sensor_test_probs.npz",
        sample_id=frame["sample_id"].astype(str).to_numpy(dtype=str),
        proba=ensemble
    )
    print("센서 확률 저장 완료", flush=True)

    pred = np.argmax(ensemble, axis=1) + 1

    output = pd.DataFrame({
        "sample_id": frame["sample_id"],
        "label": pred.astype(int),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"저장 완료: {args.output} ({len(output)}행)")


if __name__ == "__main__":
    main()
