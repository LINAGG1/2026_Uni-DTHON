
import numpy as np
import pandas as pd

from sensor_features import build_sensor_features
from model import numeric, add_causal_features


def build_sensor_features_v3(frame):
    # 기존 센서 특징 유지
    original = build_sensor_features(frame)

    base = frame[["experiment_id", "relative_time_s"]].copy()
    base["relative_time_s"] = pd.to_numeric(
        base["relative_time_s"], errors="coerce"
    )

    sensors = [
        "meas_t1", "meas_t2", "thermal_t",
        "meas_p", "meas_co", "meas_co2",
        "meas_hf", "atmp_t",
        "bat_cc", "soc_status", "tr_method"
    ]
    base[sensors] = numeric(frame, sensors)

    # 시간순으로 정렬하되 원래 행 순서는 유지
    ordered = base.sort_values(
        ["experiment_id", "relative_time_s"],
        kind="mergesort"
    ).copy()

    group = ordered.groupby("experiment_id", sort=False)

    dt = group["relative_time_s"].diff()
    dt = dt.where(dt > 0)

    # 1. 온도·압력·가스의 직전 관측 대비 변화율
    rate_cols = [
        "meas_t1", "meas_t2", "thermal_t",
        "meas_p", "meas_co", "meas_co2", "meas_hf"
    ]

    for col in rate_cols:
        ordered[f"{col}_rate"] = group[col].diff() / dt

    # 2. 온도 상승 가속도
    for col in ["meas_t1", "meas_t2", "thermal_t"]:
        rate_col = f"{col}_rate"
        ordered[f"{col}_accel"] = (
            ordered.groupby("experiment_id", sort=False)[rate_col].diff()
            / dt
        )

    # 3. 압력과 가스의 동시 변화
    ordered["pressure_co_coupling"] = (
        ordered["meas_p_rate"] * ordered["meas_co_rate"]
    )
    ordered["pressure_co2_coupling"] = (
        ordered["meas_p_rate"] * ordered["meas_co2_rate"]
    )

    # 4. 온도 상승과 가스 발생의 동시 변화
    ordered["temp_co_coupling"] = (
        ordered["meas_t1_rate"] * ordered["meas_co_rate"]
    )

    # 5. 실험 조건과 온도 변화의 상호작용
    ordered["temp_rate_soc"] = (
        ordered["meas_t1_rate"] * ordered["soc_status"]
    )
    ordered["temp_rate_capacity"] = (
        ordered["meas_t1_rate"] * np.log1p(
            ordered["bat_cc"].clip(lower=0)
        )
    )

    # 6. 현재까지 관측된 고온 지속 시간 (고정 임계값 실험용)
    for threshold in [80, 120]:
        hot = ordered["meas_t1"].ge(threshold).astype(float)
        ordered[f"hot_duration_{threshold}"] = (
            (hot * dt.fillna(0))
            .groupby(ordered["experiment_id"]).cumsum()
        )

    # 7. 변화율의 과거 30초 통계
    rolling_signals = [
        "meas_t1_rate", "meas_t2_rate",
        "thermal_t_rate", "meas_p_rate",
        "meas_co_rate", "meas_co2_rate"
    ]

    rolling = add_causal_features(
        ordered,
        rolling_signals,
        window_s=30.0
    )

    new_columns = [
        col for col in ordered.columns
        if col.endswith("_rate")
        or col.endswith("_accel")
        or col.endswith("_coupling")
        or col.startswith("temp_rate_")
        or col.startswith("hot_duration_")
    ]

    # 원래 입력 행 순서로 복원
    new_features = ordered[new_columns].reindex(base.index)
    rolling = rolling.reindex(base.index)

    result = pd.concat(
        [
            original.reset_index(drop=True),
            new_features.reset_index(drop=True),
            rolling.reset_index(drop=True)
        ],
        axis=1
    )

    result = result.replace([np.inf, -np.inf], np.nan)

    assert result.columns.is_unique
    assert len(result) == len(frame)

    return result
