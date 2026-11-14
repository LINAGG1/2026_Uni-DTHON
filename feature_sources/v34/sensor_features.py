
"""센서 및 과거 시계열 특징 생성."""

import numpy as np
import pandas as pd

from .model import add_causal_features, numeric


SENSOR_COLUMNS = [
    "meas_v", "meas_t1", "meas_t2", "meas_p",
    "meas_ch4", "meas_co", "meas_co2",
    "meas_hcn", "meas_hcl", "meas_hf",
    "meas_n2o", "meas_no", "meas_no2", "meas_so2",
    "atmp_t", "thermal_t", "indu_t",
]

CONDITION_COLUMNS = [
    "bat_class", "bat_cc", "soc_status", "tr_method",
]

WINDOWS = [10.0, 30.0, 60.0]


def build_sensor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """현재 및 같은 실험의 과거 관측만 사용합니다."""

    base = frame[["experiment_id", "relative_time_s"]].copy()

    base["relative_time_s"] = pd.to_numeric(
        base["relative_time_s"], errors="coerce"
    )

    base[SENSOR_COLUMNS] = numeric(frame, SENSOR_COLUMNS)
    base[CONDITION_COLUMNS] = numeric(frame, CONDITION_COLUMNS)

    # 온도 차이
    base["temp_diff"] = base["meas_t1"] - base["meas_t2"]
    base["temp_surface_ambient"] = base["meas_t1"] - base["atmp_t"]
    base["temp_thermal_ambient"] = base["thermal_t"] - base["atmp_t"]

    # 온도 합계
    base["temp_sum"] = base["meas_t1"] + base["meas_t2"]

    # 과거 윈도우 특징
    signals = [
        "meas_v", "meas_t1", "meas_t2", "meas_p",
        "meas_co", "meas_co2", "meas_hf",
        "thermal_t", "temp_diff",
    ]

    features = [
        base[
            ["relative_time_s"]
            + SENSOR_COLUMNS
            + CONDITION_COLUMNS
            + ["temp_diff", "temp_surface_ambient",
               "temp_thermal_ambient", "temp_sum"]
        ]
    ]

    for window in WINDOWS:
        causal = add_causal_features(
            base, signals, window_s=window
        )
        features.append(causal)

    result = pd.concat(features, axis=1)
    return result.replace([np.inf, -np.inf], np.nan)
