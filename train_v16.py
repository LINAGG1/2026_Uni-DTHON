from pathlib import Path
import copy
import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from train import load_split
from sensor_features_v3 import build_sensor_features_v3

DATA_DIR = Path.home() / "data/battery_data_light"
BASE_PATH = Path("weights/model_sensor_v7_gpu.joblib")
OUT_PATH = Path("weights/model_sensor_v16_cnn.joblib")

LABELS = np.arange(1, 7)
SEQ_LEN = 64
WINDOW_SECONDS = 60
BATCH_SIZE = 256
EPOCHS = 30
PATIENCE = 6
SEED = 42

SENSORS = [
    "meas_v", "meas_a", "meas_t1", "meas_t2",
    "meas_p", "meas_ch4", "meas_co", "meas_co2",
    "meas_hcn", "meas_hcl", "meas_hf",
    "meas_n2o", "meas_no", "meas_no2",
    "meas_so2", "atmp_t", "thermal_t",
]

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device, flush=True)


def load_data(split):
    frame, X_base, y = load_split(DATA_DIR, split)

    X_sensor = build_sensor_features_v3(frame)
    X_sensor = X_sensor.drop(
        columns=X_base.columns.intersection(X_sensor.columns),
        errors="ignore"
    )

    X = pd.concat(
        [
            X_base.reset_index(drop=True),
            X_sensor.reset_index(drop=True),
        ],
        axis=1,
    )

    return frame.reset_index(drop=True), X, y.reset_index(drop=True)


def make_sequences(frame, scaler):
    """각 실험에서 현재 시점까지 최대 60초, 최대 64개 관측값."""
    n = len(frame)
    sequences = np.zeros(
        (n, SEQ_LEN, len(SENSORS) + 1),
        dtype=np.float32,
    )

    sensor_values = (
        frame[SENSORS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )

    sensor_values = scaler.transform(sensor_values)
    sensor_values = np.clip(sensor_values, -10, 10)

    times = pd.to_numeric(
        frame["relative_time_s"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    for _, indices in frame.groupby(
        "experiment_id", sort=False
    ).indices.items():
        indices = np.asarray(indices)
        order = np.argsort(times[indices], kind="stable")
        idx = indices[order]
        t = times[idx]

        for j, row_idx in enumerate(idx):
            start = np.searchsorted(
                t, t[j] - WINDOW_SECONDS, side="left"
            )
            start = max(start, j - SEQ_LEN + 1)

            history_idx = idx[start:j + 1]
            length = len(history_idx)

            sequences[row_idx, -length:, :-1] = (
                sensor_values[history_idx]
            )
            sequences[row_idx, -length:, -1] = 1.0

    return sequences


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(
            np.asarray(y, dtype=np.int64) - 1
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SensorCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2))


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    outputs = []

    for start in range(0, len(X), BATCH_SIZE):
        batch = torch.from_numpy(
            X[start:start + BATCH_SIZE]
        ).to(device)

        logits = model(batch)
        proba = torch.softmax(logits, dim=1)
        outputs.append(proba.cpu().numpy())

    return np.concatenate(outputs)


def score(y, pred):
    return f1_score(
        y, pred, labels=LABELS,
        average="macro", zero_division=0
    )


print("Train / Validation 로딩 중...", flush=True)
train_frame, X_train_base, y_train = load_data("train")
val_frame, X_val_base, y_val = load_data("validation")

# 센서 정규화는 Train 데이터만 사용
train_raw = (
    train_frame[SENSORS]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .to_numpy(dtype=np.float64)
)

scaler = StandardScaler()
scaler.fit(train_raw)

print("시계열 구성 중...", flush=True)
X_train_seq = make_sequences(train_frame, scaler)
X_val_seq = make_sequences(val_frame, scaler)

print("Train:", X_train_seq.shape)
print("Validation:", X_val_seq.shape)

train_loader = DataLoader(
    SequenceDataset(X_train_seq, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

model = SensorCNN(len(SENSORS) + 1).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.001,
)

# 기존 모델과 동일한 클래스 가중치
weights = torch.tensor(
    [1, 1, 1, 0.7, 1.5, 1],
    dtype=torch.float32,
    device=device,
)

criterion = nn.CrossEntropyLoss(weight=weights)

best_score = -1
best_state = None
bad_epochs = 0

print("\n16차 1D-CNN 학습 중...", flush=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_proba = predict_proba(model, X_val_seq)
    val_pred = LABELS[np.argmax(val_proba, axis=1)]
    val_score = score(y_val, val_pred)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Loss {total_loss/len(train_loader):.4f} | "
        f"Val Macro F1 {val_score:.6f}",
        flush=True,
    )

    if val_score > best_score:
        best_score = val_score
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
    else:
        bad_epochs += 1

    if bad_epochs >= PATIENCE:
        print("Early stopping", flush=True)
        break

model.load_state_dict(best_state)

cnn_proba = predict_proba(model, X_val_seq)
cnn_pred = LABELS[np.argmax(cnn_proba, axis=1)]

# 기존 CatBoost 모델
artifact = joblib.load(BASE_PATH)
base_model = artifact["pipeline"]
features = artifact["features"]

X_val_base = X_val_base[features]

cat_raw = base_model.predict_proba(X_val_base)
cat_classes = np.asarray(base_model.classes_).astype(int)

cat_proba = np.column_stack([
    cat_raw[:, np.where(cat_classes == c)[0][0]]
    for c in LABELS
])

cat_pred = LABELS[np.argmax(cat_proba, axis=1)]

print("\n=== 16차 결과 ===")
print(f"6.1차 CatBoost: {score(y_val, cat_pred):.6f}")
print(f"1D-CNN 단독: {score(y_val, cnn_pred):.6f}")

results = []

for cat_weight in [0.25, 0.50, 0.75, 0.90]:
    ensemble = (
        cat_weight * cat_proba
        + (1 - cat_weight) * cnn_proba
    )

    pred = LABELS[np.argmax(ensemble, axis=1)]
    value = score(y_val, pred)

    results.append((cat_weight, value))

    print(
        f"앙상블 CatBoost {cat_weight:.0%}: "
        f"{value:.6f}"
    )

best_weight, ensemble_score = max(
    results, key=lambda x: x[1]
)

best_ensemble = (
    best_weight * cat_proba
    + (1 - best_weight) * cnn_proba
)

final_pred = LABELS[np.argmax(best_ensemble, axis=1)]

print("\n최고 앙상블 비율:", best_weight)
print("최고 앙상블 F1:", ensemble_score)

print("\n=== 앙상블 단계별 결과 ===")
print(classification_report(
    y_val, final_pred, labels=LABELS, digits=4
))

print("\n=== 혼동행렬 ===")
print(confusion_matrix(
    y_val, final_pred, labels=LABELS
))

OUT_PATH.parent.mkdir(exist_ok=True)

joblib.dump(
    {
        "state_dict": {
            k: v.cpu()
            for k, v in model.state_dict().items()
        },
        "scaler": scaler,
        "sensors": SENSORS,
        "seq_len": SEQ_LEN,
        "window_seconds": WINDOW_SECONDS,
        "best_epoch_val_f1": best_score,
        "best_cat_weight": best_weight,
        "feature_version": "sensor_cnn_v16",
    },
    OUT_PATH,
)

print("\n16차 모델 저장 완료!", flush=True)
