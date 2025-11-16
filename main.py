#!/usr/bin/env python3
"""
Improved training pipeline for the chunchunmaru-AI project.

Key ideas:
- Build temporal features (including autoregressive Light_Kitchen lags) on the
  concatenated train+test timeline to avoid gaps at day boundaries.
- Train a PyTorch feedforward network on scaled features.
- Evaluate with a chronological split and sequential (teacher-forced) rollout
  that mimics the test-time setup where past Light_Kitchen values are unknown.
- Use the best validation threshold for the final submission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Base sensor columns present both in train and test.
BASE_SENSOR_COLUMNS: Sequence[str] = [
    "TimeOfDay",
    "month",
    "Movement_Living",
    "Movement_Kitchen",
    "Movement_Hallway",
    "FrontDoor_Open",
    "Light_Living",
    "Kettle_On",
    "Heater_State",
    "Window_State",
    "Power",
    "Temperature",
    "Humidity",
]

TARGET_COLUMN = "Light_Kitchen"
VAL_FRACTION = 0.1
LAG_STEPS = [1, 5, 10, 30, 60]
ROLLING_WINDOWS = [30, 120, 600]
TARGET_LAG_STEPS = [1, 2, 5, 10, 30, 60]
LAG_NOISE_PROB = 0.05

# Training hyperparameters for the PyTorch model.
TRAIN_EPOCHS = 15
BATCH_SIZE = 4096
LEARNING_RATE = 3e-4
PATIENCE = 3
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chunchunmaru-AI model and create submission.")
    parser.add_argument("--train-path", default="train_data.csv", type=Path, help="Path to the training data CSV.")
    parser.add_argument("--test-path", default="test_data.csv", type=Path, help="Path to the test data CSV.")
    parser.add_argument("--output-path", default="submission.csv", type=Path, help="Where to save predictions.")
    return parser.parse_args()


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eda(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Basic sanity checks printed to stdout."""
    print("===== EDA =====")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"Train columns: {list(train_df.columns)}")
    print(f"Test columns:  {list(test_df.columns)}")
    print("\nTrain describe (numeric columns):")
    print(train_df.describe().T)
    target_counts = train_df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    print("\nTarget distribution:")
    print(target_counts)
    print("Target share:")
    print((target_counts / target_counts.sum()).round(4))
    print("\nMissing values in train (top 10):")
    print(train_df.isna().sum().sort_values(ascending=False).head(10))
    print("\nMissing values in test (top 10):")
    print(test_df.isna().sum().sort_values(ascending=False).head(10))
    print("================\n")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds calendar/time-based columns while keeping chronological order."""
    result = df.sort_values("timestamp").reset_index(drop=True).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["hour"] = result["timestamp"].dt.hour
    result["minute"] = result["timestamp"].dt.minute
    result["second"] = result["timestamp"].dt.second
    result["dayofweek"] = result["timestamp"].dt.dayofweek
    result["is_weekend"] = (result["dayofweek"] >= 5).astype(int)
    result["seconds_from_midnight"] = result["hour"] * 3600 + result["minute"] * 60 + result["second"]
    result["seconds_from_start"] = (result["timestamp"] - result["timestamp"].iloc[0]).dt.total_seconds()
    result["sin_hour"] = np.sin(2 * np.pi * result["hour"] / 24.0)
    result["cos_hour"] = np.cos(2 * np.pi * result["hour"] / 24.0)
    return result


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    lag_steps: Sequence[int],
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    """
    Create lagged values and trailing rolling means.

    pandas shift/rolling primitives rely only on past observations, so there is no leakage.
    """
    result = df.copy()
    engineered: dict[str, pd.Series] = {}
    available_cols = [col for col in columns if col in result.columns]
    for col in available_cols:
        for lag in lag_steps:
            engineered[f"{col}_lag_{lag}"] = result[col].shift(lag)
        for window in rolling_windows:
            engineered[f"{col}_roll_mean_{window}"] = result[col].rolling(window=window, min_periods=1).mean()
    if engineered:
        engineered_df = pd.DataFrame(engineered)
        result = pd.concat([result, engineered_df], axis=1)
    return result


def build_feature_table(
    df: pd.DataFrame,
    sensor_cols: Sequence[str],
    lag_steps: Sequence[int],
    rolling_windows: Sequence[int],
) -> pd.DataFrame:
    """Wrapper combining the time features with lag/rolling statistics."""
    processed = add_time_features(df)
    processed = add_lag_and_rolling_features(processed, sensor_cols, lag_steps, rolling_windows)
    return processed


def add_target_lags(df: pd.DataFrame, target: str, lag_steps: Sequence[int]) -> pd.DataFrame:
    """Adds shifted target columns (historical Light_Kitchen values)."""
    result = df.copy()
    for lag in lag_steps:
        result[f"{target}_lag_{lag}"] = result[target].shift(lag)
    return result


def chronological_split_idx(length: int, val_fraction: float) -> int:
    split_idx = int(length * (1.0 - val_fraction))
    split_idx = max(1, min(split_idx, length - 1))
    return split_idx


def build_history_seed(series: pd.Series, lag_steps: Sequence[int]) -> List[float]:
    """Extracts the last max(lag_steps) Light_Kitchen values to warm-start autoregressive inference."""
    if not lag_steps:
        return []
    max_lag = max(lag_steps)
    values = series.astype(float).tolist()
    if len(values) >= max_lag:
        return values[-max_lag:]
    if not values:
        return [0.0] * max_lag
    pad_value = values[0]
    padding = [pad_value] * (max_lag - len(values))
    return padding + values


def inject_lag_noise(
    df: pd.DataFrame,
    lag_columns: Sequence[str],
    noise_prob: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Randomly flips some lag values to mimic imperfect history during training."""
    if not lag_columns or noise_prob <= 0:
        return df
    rng = np.random.default_rng(seed)
    result = df.copy()
    for col in lag_columns:
        if col not in result.columns:
            continue
        values = result[col].to_numpy(dtype=float)
        mask = rng.random(len(values)) < noise_prob
        values[mask] = 1.0 - values[mask]
        result[col] = values
    return result


class TabularNet(nn.Module):
    """Simple feedforward network for tabular classification."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
) -> Tuple[nn.Module, dict]:
    """Trains the neural net and returns it together with training metadata."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularNet(x_train.shape[1]).to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    positive = float(y_train.sum())
    negative = float(len(y_train) - positive)
    pos_weight = negative / (positive + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    best_state = None
    best_epoch = 0
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_dataset)
        log_msg = f"Epoch {epoch + 1}/{epochs} train_loss={avg_train_loss:.5f}"

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.from_numpy(x_val.astype(np.float32)).to(device))
                val_loss = criterion(val_logits, torch.from_numpy(y_val.astype(np.float32)).to(device)).item()
            log_msg += f" val_loss={val_loss:.5f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(log_msg + " (early stop)")
                    break
        else:
            best_epoch = epoch + 1
        print(log_msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_epoch": max(best_epoch, 1)}


def torch_predict_proba(model: nn.Module, data: np.ndarray) -> np.ndarray:
    """Returns sigmoid probabilities for a batch of samples."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(data.astype(np.float32)).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def sequential_probability_rollout(
    model: nn.Module,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
    lag_columns: Sequence[str],
    lag_steps: Sequence[int],
    history_seed: Sequence[float],
    decision_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts probabilities row-by-row, feeding the model with its own past outputs.

    This mimics the test setting where Light_Kitchen history for the current day is unknown.
    """
    if not len(features_df):
        return np.array([], dtype=float), np.array([], dtype=int)

    arr = features_df.to_numpy(dtype=np.float32, copy=True)
    lag_indices = [features_df.columns.get_loc(col) for col in lag_columns]
    history = [float(x) for x in history_seed]
    device = next(model.parameters()).device
    probs = np.zeros(len(arr), dtype=np.float32)
    pred_classes = np.zeros(len(arr), dtype=int)

    model.eval()
    for idx in range(len(arr)):
        for col_idx, lag in zip(lag_indices, lag_steps):
            if lag <= len(history):
                arr[idx, col_idx] = history[-lag]
            elif history:
                arr[idx, col_idx] = history[0]
            else:
                arr[idx, col_idx] = 0.0
        scaled_row = scaler.transform(arr[idx : idx + 1])
        with torch.no_grad():
            logit = model(torch.from_numpy(scaled_row.astype(np.float32)).to(device))
            prob = torch.sigmoid(logit).item()
        probs[idx] = prob
        pred_class = 1 if prob >= decision_threshold else 0
        pred_classes[idx] = pred_class
        history.append(float(pred_class))
    return probs, pred_classes


def main() -> None:
    args = parse_args()
    set_seed()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    run_eda(train_df, test_df)

    # Prepare combined dataframe to ensure lag features use history across the 8->9 day boundary.
    train_aug = train_df.copy()
    train_aug["dataset"] = "train"
    train_aug["ID"] = -1  # placeholder to align with test columns

    test_aug = test_df.copy()
    test_aug["dataset"] = "test"
    test_aug[TARGET_COLUMN] = np.nan  # placeholder for compatibility

    combined_df = pd.concat([train_aug, test_aug], ignore_index=True, sort=False)
    sensor_columns = [col for col in BASE_SENSOR_COLUMNS if col in combined_df.columns]
    processed_combined = build_feature_table(combined_df, sensor_columns, LAG_STEPS, ROLLING_WINDOWS)

    processed_train = (
        processed_combined[processed_combined["dataset"] == "train"]
        .reset_index(drop=True)
        .drop(columns=["dataset"])
    )
    processed_test = (
        processed_combined[processed_combined["dataset"] == "test"]
        .reset_index(drop=True)
        .drop(columns=["dataset"])
    )

    processed_train = add_target_lags(processed_train, TARGET_COLUMN, TARGET_LAG_STEPS)
    lag_feature_columns = [f"{TARGET_COLUMN}_lag_{lag}" for lag in TARGET_LAG_STEPS]
    processed_train = processed_train.dropna(subset=lag_feature_columns).reset_index(drop=True)
    processed_train = inject_lag_noise(processed_train, lag_feature_columns, LAG_NOISE_PROB, RANDOM_SEED)

    # Ensure the test table contains the lag columns, which will be filled sequentially later.
    for lag_col in lag_feature_columns:
        processed_test[lag_col] = np.nan

    test_ids_ordered = processed_test["ID"].astype(int).reset_index(drop=True)
    processed_train = processed_train.drop(columns=["ID"])
    processed_test = processed_test.drop(columns=["ID"])

    feature_columns = [
        col
        for col in processed_train.columns
        if col not in {TARGET_COLUMN, "timestamp"}
    ]

    split_idx = chronological_split_idx(len(processed_train), VAL_FRACTION)
    train_slice = processed_train.iloc[:split_idx].reset_index(drop=True)
    val_slice = processed_train.iloc[split_idx:].reset_index(drop=True)

    x_train_df = train_slice[feature_columns]
    x_val_df = val_slice[feature_columns]
    train_medians = x_train_df.median()
    x_train = x_train_df.fillna(train_medians).to_numpy(dtype=np.float32)
    x_val = x_val_df.fillna(train_medians).to_numpy(dtype=np.float32)
    y_train = train_slice[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val_slice[TARGET_COLUMN].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    model, history = train_torch_model(x_train_scaled, y_train, x_val_scaled, y_val)

    # Classic evaluation (uses actual target lags) for reference.
    val_probs = torch_predict_proba(model, x_val_scaled)
    val_preds = (val_probs >= 0.5).astype(int)
    print(
        f"Validation (teacher forcing) accuracy={accuracy_score(y_val, val_preds):.4f} "
        f"f1={f1_score(y_val, val_preds):.4f}"
    )

    # Sequential evaluation that mirrors the test workflow.
    val_history_seed = build_history_seed(train_slice[TARGET_COLUMN], TARGET_LAG_STEPS)
    val_features_seq = val_slice[feature_columns].fillna(train_medians)

    # Evaluate sequential rollout for a range of decision thresholds.
    candidate_thresholds = [0.05, 0.10, 0.15, 0.20]
    best_thr = 0.5
    best_f1 = -1.0
    best_probs_seq = None
    best_preds_seq = None
    default_probs_seq, default_preds_seq = sequential_probability_rollout(
        model,
        scaler,
        val_features_seq,
        lag_feature_columns,
        TARGET_LAG_STEPS,
        val_history_seed,
        decision_threshold=0.5,
    )
    default_acc = accuracy_score(y_val, default_preds_seq)
    default_f1 = f1_score(y_val, default_preds_seq)
    for thr in candidate_thresholds:
        probs_seq, preds_seq = sequential_probability_rollout(
            model,
            scaler,
            val_features_seq,
            lag_feature_columns,
            TARGET_LAG_STEPS,
            val_history_seed,
            decision_threshold=float(thr),
        )
        seq_f1 = f1_score(y_val, preds_seq)
        if seq_f1 > best_f1:
            best_f1 = seq_f1
            best_thr = float(thr)
            best_probs_seq = probs_seq
            best_preds_seq = preds_seq
    print(
        f"Sequential validation (thr=0.5) accuracy={default_acc:.4f} f1={default_f1:.4f}"
    )
    print(
        f"Best sequential threshold={best_thr:.3f} accuracy={accuracy_score(y_val, best_preds_seq):.4f} "
        f"f1={best_f1:.4f}"
    )

    # Refit scaler/model on the full training data for the final submission.
    x_full_df = processed_train[feature_columns]
    full_medians = x_full_df.median()
    x_full = x_full_df.fillna(full_medians).to_numpy(dtype=np.float32)
    y_full = processed_train[TARGET_COLUMN].to_numpy(dtype=np.float32)
    full_scaler = StandardScaler()
    x_full_scaled = full_scaler.fit_transform(x_full)
    final_epochs = max(history["best_epoch"], 5)
    final_model, _ = train_torch_model(
        x_full_scaled,
        y_full,
        x_val=None,
        y_val=None,
        epochs=final_epochs,
    )

    # Sequential rollout on the test day using the full model.
    test_features = processed_test[feature_columns].fillna(full_medians)
    test_history_seed = build_history_seed(processed_train[TARGET_COLUMN], TARGET_LAG_STEPS)
    test_probs, test_pred_classes = sequential_probability_rollout(
        final_model,
        full_scaler,
        test_features,
        lag_feature_columns,
        TARGET_LAG_STEPS,
        test_history_seed,
        decision_threshold=best_thr,
    )

    submission = pd.DataFrame(
        {
            "ID": test_ids_ordered,
            TARGET_COLUMN: test_pred_classes,
        }
    ).sort_values("ID")
    submission.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path.resolve()}")
    print(f"Decision threshold used for submission: {best_thr:.3f}")


if __name__ == "__main__":
    main()
