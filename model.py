"""Feature preparation and probability combination for the Uni-D submission."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


LABELS = np.arange(1, 7)
REFERENCE_SENSORS = (
    "meas_v", "meas_a", "meas_t1", "meas_t2", "thermal_t", "meas_p",
    "meas_co", "meas_co2", "meas_hf",
)
CATEGORICAL_COLUMNS = {"bat_class", "bat_cc", "soc_status", "tr_method"}


def build_submission_features(frame, image_root, *, raw_v34=False):
    """Build each branch's inputs from the requested rows and their images."""
    from feature_sources import event_history
    from feature_sources.rgb13 import model as rgb_model
    from feature_sources.v34 import model as common_model
    from feature_sources.v34.sensor_features_v34 import build_sensor_features_v34

    frame = frame.reset_index(drop=True)
    roots = [image_root]
    if image_root.parent != image_root:
        roots.append(image_root.parent)
    rgb_paths = rgb_model.resolve_image_paths(frame, roots)
    rgb_images = rgb_model.extract_image_features(frame, rgb_paths)
    rgb = rgb_model.build_features(frame, rgb_images)
    if rgb.index.name == "sample_id" and not np.array_equal(rgb.index, frame["sample_id"]):
        raise ValueError("13차 특징의 sample_id 순서가 다릅니다.")

    common_paths = common_model.resolve_image_paths(frame, roots)
    if raw_v34:
        images = common_model.extract_image_features(frame, common_paths)
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            values = list(pool.map(common_model._image_stats, common_paths))
        images = pd.DataFrame(
            values, columns=common_model.IMAGE_STATS,
            index=pd.Index(frame["sample_id"].values, name="sample_id"),
        )
    base = common_model.build_features(frame, images).reset_index(drop=True)
    extra = build_sensor_features_v34(frame).reset_index(drop=True)
    common = pd.concat(
        [base, extra.drop(columns=base.columns.intersection(extra.columns))], axis=1
    )
    if not raw_v34:
        common = prepare_common_features(common)
    events = event_history.event_features(frame)
    if rgb.shape != (len(frame), 49) or common.shape != (len(frame), 176):
        raise ValueError(f"13차 / 34차 특징 크기 확인 필요: {rgb.shape} / {common.shape}")
    return rgb, common, events


def prepare_common_features(common):
    """Use the sentinel expected by the other branches sharing v34 features."""
    return common.replace([np.inf, -np.inf], np.nan).fillna(-999)


def make_lightgbm_features(frame, v34_features):
    """Add the 27 reference features used by the saved leaves15 models."""
    if len(frame) != len(v34_features):
        raise ValueError("CSV 행과 34차 특징의 행 수가 다릅니다.")
    signals = {
        name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        for name in REFERENCE_SENSORS
    }
    reference = {
        f"reference_{name}_{stat}": np.full(len(frame), np.nan)
        for name in REFERENCE_SENSORS
        for stat in ("initial_delta", "past_min_delta", "past_max_delta")
    }
    time = frame["relative_time_s"].to_numpy()
    for positions in frame.groupby("experiment_id", sort=False).indices.values():
        positions = positions[np.argsort(time[positions], kind="stable")]
        for name, signal in signals.items():
            values = signal[positions]
            series = pd.Series(values)
            reference[f"reference_{name}_initial_delta"][positions] = values - values[0]
            reference[f"reference_{name}_past_min_delta"][positions] = (
                values - series.cummin().to_numpy()
            )
            reference[f"reference_{name}_past_max_delta"][positions] = (
                values - series.cummax().to_numpy()
            )
    base = v34_features.reset_index(drop=True).replace(-999, np.nan)
    return pd.concat([base, pd.DataFrame(reference)], axis=1)


def predict_lightgbm(model, features):
    """Apply the saved feature order and category encoding for one fold."""
    columns = model.feature_name_
    categories = [name for name in columns if name in CATEGORICAL_COLUMNS]
    levels = model.booster_.pandas_categorical
    if levels is None or len(categories) != len(levels):
        raise ValueError("LightGBM 가중치에 저장된 범주 인코딩을 확인하세요.")
    if not np.array_equal(model.classes_, LABELS):
        raise ValueError("LightGBM 클래스 순서가 1~6과 다릅니다.")

    X = features.loc[:, columns].copy()
    for name, values in zip(categories, levels):
        X[name] = pd.Categorical(X[name], categories=values)
    return model.predict_proba(X)


def make_extratrees_features(v34_features, event_features, lightgbm_model):
    """Apply the category coding and numeric conversion used for ExtraTrees."""
    base = v34_features.reset_index(drop=True).replace(-999, np.nan)
    events = event_features.reset_index(drop=True).add_prefix("event_")
    if len(base) != len(events):
        raise ValueError("34차 특징과 이력 특징의 행 수가 다릅니다.")
    X = pd.concat([base, events], axis=1)
    if not X.columns.is_unique or X.shape[1] != 797:
        raise ValueError(f"ExtraTrees 입력 특징 확인 필요: {X.shape[1]}개")
    categories = [
        name for name in lightgbm_model.feature_name_ if name in CATEGORICAL_COLUMNS
    ]
    levels = lightgbm_model.booster_.pandas_categorical
    if levels is None or len(categories) != len(levels):
        raise ValueError("가중치에 저장된 범주 인코딩을 확인하세요.")
    for name, values in zip(categories, levels):
        X[name] = pd.Categorical(X[name], categories=values).codes.astype(float)
    return X.fillna(-999).astype(np.float32)


def redistribute(probabilities, labels, conditional, strength):
    """Change the distribution within the given classes, keeping their total."""
    selected = np.asarray(labels, dtype=int) - 1
    adjusted = probabilities.copy()
    mass = adjusted[:, selected].sum(axis=1, keepdims=True)
    original = adjusted[:, selected] / np.maximum(mass, 1e-12)
    adjusted[:, selected] = mass * (
        (1 - strength) * original + strength * conditional
    )
    return adjusted


def combine_probabilities(v34, stage23_p3, rgb13, leaves15, expert456):
    """Combine aligned rows, with class columns in ascending label order."""
    n_rows = len(v34)
    expected_shapes = (
        ("v34", v34, (n_rows, 6)),
        ("stage23_p3", stage23_p3, (n_rows,)),
        ("rgb13", rgb13, (n_rows, 6)),
        ("leaves15", leaves15, (n_rows, 6)),
        ("expert456", expert456, (n_rows, 3)),
    )
    for name, values, shape in expected_shapes:
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite probabilities with shape {shape}")

    # Match submit_stage23.py: its conditional class 3 score is complemented
    # after interpolation to obtain the class 2 score.
    stage23 = v34.copy()
    mass23 = stage23[:, 1] + stage23[:, 2]
    original_p3 = stage23[:, 2] / np.maximum(mass23, 1e-12)
    adjusted_p3 = 0.25 * original_p3 + 0.75 * stage23_p3
    stage23[:, 1] = mass23 * (1 - adjusted_p3)
    stage23[:, 2] = mass23 * adjusted_p3
    stage23_b = 0.50 * stage23 + 0.50 * rgb13
    base = 0.90 * stage23_b + 0.10 * leaves15
    return redistribute(base, (4, 5, 6), expert456, strength=0.75)
