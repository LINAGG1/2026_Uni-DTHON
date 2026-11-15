import numpy as np
import pandas as pd

from .sensor_features_v3 import build_sensor_features_v3


def build_sensor_features_v34(frame):
    original = build_sensor_features_v3(frame)

    sensors = [
        "meas_t1", "meas_t2", "thermal_t",
        "meas_p", "meas_co", "meas_co2",
    ]

    base = frame[
        ["experiment_id", "relative_time_s"] + sensors
    ].copy()

    base["relative_time_s"] = pd.to_numeric(
        base["relative_time_s"], errors="coerce"
    )

    for col in sensors:
        base[col] = pd.to_numeric(
            base[col], errors="coerce"
        )

    base["_original_index"] = np.arange(len(base))

    base = base.sort_values(
        ["experiment_id", "relative_time_s", "_original_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    groups = base.groupby("experiment_id", sort=False)

    # 과거 최대값
    for col in ["meas_t1", "meas_t2", "thermal_t"]:
        base[f"{col}_past_max"] = groups[col].cummax()
        base[f"{col}_below_past_max"] = (
            base[f"{col}_past_max"] - base[col]
        )

    # 10초, 60초 이전 관측값 대비 변화량
    for window in [10.0, 60.0]:
        for col in sensors:
            base[f"{col}_delta_{int(window)}s"] = np.nan

        for _, idx in base.groupby(
            "experiment_id", sort=False
        ).indices.items():

            idx = np.asarray(idx)
            t = base.loc[idx, "relative_time_s"].to_numpy(float)

            past = np.searchsorted(
                t, t - window, side="right"
            ) - 1

            valid = past >= 0

            for col in sensors:
                values = base.loc[idx, col].to_numpy(float)
                delta = np.full(len(idx), np.nan)

                delta[valid] = (
                    values[valid] - values[past[valid]]
                )

                base.loc[
                    idx, f"{col}_delta_{int(window)}s"
                ] = delta

    new_cols = [
        c for c in base.columns
        if "_past_max" in c
        or "_below_past_max" in c
        or "_delta_" in c
    ]

    extra = (
        base.sort_values("_original_index")[new_cols]
        .reset_index(drop=True)
    )

    result = pd.concat(
        [original.reset_index(drop=True), extra],
        axis=1,
    )

    result = result.replace([np.inf, -np.inf], np.nan)

    assert result.columns.is_unique
    assert len(result) == len(frame)

    return result
