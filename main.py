"""
Smart meter forecasting + solar/battery/grid simulation

Run:
  python main.py

Notes:
- Expects `household_power.csv` in the same folder as this script.
- Dataset format detected from file header:
    Date;Time;Global_active_power;...
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def _ensure_tensorflow() -> None:
    """
    Ensure TensorFlow is importable.

    This program is meant to be runnable as `python main.py`. If TensorFlow is
    missing, we attempt a one-time non-interactive install.
    """
    # TensorFlow wheels are typically only available for specific Python versions.
    # If you're on a bleeding-edge Python, installing TF via pip will fail.
    if sys.version_info >= (3, 13):
        raise RuntimeError(
            "TensorFlow is required for the LSTM model, but it is not available for "
            f"your Python version ({sys.version.split()[0]}). "
            "Create a Python 3.11 or 3.12 environment, then install dependencies:\n"
            "  python -m pip install tensorflow pandas numpy scikit-learn matplotlib\n"
            "Then rerun:\n"
            "  python main.py"
        )

    try:
        import tensorflow as _  # noqa: F401
        return
    except Exception:
        pass

    import subprocess

    print("TensorFlow not found. Installing it now (this may take a few minutes)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])


_ensure_tensorflow()

# TensorFlow is now expected to import successfully.
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping


@dataclass
class Config:
    # Data
    csv_path: str = "household_power.csv"
    datetime_col_name: str = "timestamp"  # created column name after parsing
    target_col: str = "Global_active_power"

    # Resampling / features
    rolling_hours: int = 24

    # LSTM windowing
    lookback_hours: int = 168  # past 1 week -> predict next hour

    # Train/test split
    train_frac: float = 0.80

    # Model/training
    epochs: int = 15
    batch_size: int = 64
    lstm_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    patience: int = 3
    seed: int = 42

    # Solar + battery simulation
    # Solar "bell curve" between 06:00-18:00 (inclusive-ish), peak in kW.
    solar_peak_kw_factor_of_train_peak: float = 0.60  # peak solar is 60% of peak train demand
    solar_sigma_hours: float = 2.5  # width of the bell curve around noon

    # Battery in kWh; with hourly steps, power kW ~= energy kWh per hour.
    battery_capacity_kwh: float = 6.0
    battery_max_charge_kw: float = 2.0
    battery_max_discharge_kw: float = 2.0


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_resample_hourly(cfg: Config) -> pd.DataFrame:
    """
    Load semicolon-separated smart meter data with Date/Time columns.
    Convert to datetime and resample to hourly means.
    """
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(
            f"Could not find `{cfg.csv_path}` in {os.getcwd()}. "
            f"Place the CSV next to main.py or update Config.csv_path."
        )

    # The file uses ';' separator. Some values may be '?' (missing).
    df = pd.read_csv(
        cfg.csv_path,
        sep=";",
        na_values=["?", "NA", "N/A", ""],
        low_memory=False,
    )

    # Build timestamp from Date + Time, typical format: dd/mm/YYYY and HH:MM:SS
    if "Date" in df.columns and "Time" in df.columns:
        ts = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        df[cfg.datetime_col_name] = ts
        df = df.drop(columns=["Date", "Time"])
    else:
        # Fallback: try to find a likely timestamp column.
        candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if not candidates:
            raise ValueError(
                "Could not find `Date`/`Time` columns or any obvious timestamp column."
            )
        df[cfg.datetime_col_name] = pd.to_datetime(df[candidates[0]], errors="coerce")

    # Ensure numeric columns parse correctly
    for col in df.columns:
        if col == cfg.datetime_col_name:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[cfg.datetime_col_name, cfg.target_col]).copy()
    df = df.sort_values(cfg.datetime_col_name)
    df = df.set_index(cfg.datetime_col_name)

    # Resample to hourly mean demand (and other columns if present)
    hourly = df.resample("1H").mean(numeric_only=True)

    # Clean: remove hours where target is missing after aggregation
    hourly = hourly.dropna(subset=[cfg.target_col]).copy()
    return hourly


def add_features(hourly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Create:
    - hour of day
    - day of week
    - 24-hour rolling average of demand
    """
    out = hourly.copy()
    out["hour_of_day"] = out.index.hour.astype(np.int16)
    out["day_of_week"] = out.index.dayofweek.astype(np.int16)  # Monday=0
    out[f"rolling_{cfg.rolling_hours}h_avg"] = (
        out[cfg.target_col].rolling(cfg.rolling_hours, min_periods=cfg.rolling_hours).mean()
    )

    # Drop initial rows where rolling average is not defined
    out = out.dropna(subset=[f"rolling_{cfg.rolling_hours}h_avg"]).copy()
    return out


def make_supervised_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Convert a time-indexed dataframe into LSTM sequences.
    X[t] uses rows [t-lookback, ..., t-1], y[t] is target at row t.
    """
    values_x = df[feature_cols].values.astype(np.float32)
    values_y = df[[target_col]].values.astype(np.float32)  # 2D for scalers

    X, y, idx = [], [], []
    for t in range(lookback, len(df)):
        X.append(values_x[t - lookback : t, :])
        y.append(values_y[t, 0])
        idx.append(df.index[t])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.DatetimeIndex(idx)


def main() -> None:
    cfg = Config()
    set_seeds(cfg.seed)

    # ---- Load + resample ----
    hourly = load_and_resample_hourly(cfg)
    data = add_features(hourly, cfg)

    # We'll forecast next-hour demand using past window of:
    # demand, hour_of_day, day_of_week, rolling_24h_avg
    rolling_col = f"rolling_{cfg.rolling_hours}h_avg"
    feature_cols = [cfg.target_col, "hour_of_day", "day_of_week", rolling_col]

    # ---- Train/test split (time-based) ----
    n = len(data)
    if n < cfg.lookback_hours + 50:
        raise ValueError(
            f"Not enough hourly samples after feature engineering. "
            f"Have {n}, need at least ~{cfg.lookback_hours + 50}."
        )

    split = int(math.floor(n * cfg.train_frac))
    train_df = data.iloc[:split].copy()
    test_df = data.iloc[split:].copy()

    # ---- Scale features (fit on train only) ----
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    train_X_raw = train_df[feature_cols].values
    test_X_raw = test_df[feature_cols].values

    # Scale X features
    x_scaler.fit(train_X_raw)
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_cols] = x_scaler.transform(train_X_raw)
    test_df_scaled[feature_cols] = x_scaler.transform(test_X_raw)

    # Scale y separately (target only) for stability
    y_scaler.fit(train_df[[cfg.target_col]].values)
    train_df_scaled[cfg.target_col] = y_scaler.transform(train_df[[cfg.target_col]].values)
    test_df_scaled[cfg.target_col] = y_scaler.transform(test_df[[cfg.target_col]].values)

    # ---- Build sequences ----
    X_train, y_train, _ = make_supervised_sequences(
        train_df_scaled, feature_cols=feature_cols, target_col=cfg.target_col, lookback=cfg.lookback_hours
    )
    X_test, y_test, test_index = make_supervised_sequences(
        test_df_scaled, feature_cols=feature_cols, target_col=cfg.target_col, lookback=cfg.lookback_hours
    )

    # Because we build sequences within each split, the first `lookback` hours
    # of the test split are not predicted. We'll align actuals accordingly.
    y_test_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # ---- LSTM model ----
    model = Sequential(
        [
            LSTM(cfg.lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(cfg.dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mse",
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks,
        shuffle=False,  # keep time order
    )

    # ---- Predict + evaluate ----
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    rmse = math.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    print("\n=== Test Set Evaluation (Next-Hour Demand) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R^2 : {r2:.4f}")

    # ---- Plot: predicted vs actual (requested) ----
    pred_series = pd.Series(y_pred, index=test_index, name="Predicted_demand_kW")
    actual_series = pd.Series(y_test_actual, index=test_index, name="Actual_demand_kW")

    plt.figure(figsize=(12, 4))
    plt.plot(actual_series.index, actual_series.values, label="Actual", linewidth=1)
    plt.plot(pred_series.index, pred_series.values, label="Predicted", linewidth=1)
    plt.title("Predicted vs Actual Demand (Test Period)")
    plt.xlabel("Time")
    plt.ylabel("Demand (kW)")
    plt.legend()
    plt.tight_layout()

    # ---- Solar generation simulation (bell curve 6AM-6PM) ----
    # Use a simple Gaussian bell centered at noon, clipped to [06, 18].
    train_peak_kw = float(train_df[cfg.target_col].max())
    solar_peak_kw = cfg.solar_peak_kw_factor_of_train_peak * train_peak_kw
    sigma = cfg.solar_sigma_hours

    hours = pred_series.index.hour.values.astype(np.int32)
    solar_kw = np.zeros(len(pred_series), dtype=np.float32)
    for i, h in enumerate(hours):
        if 6 <= h <= 18:
            solar_kw[i] = solar_peak_kw * math.exp(-0.5 * ((h - 12) / sigma) ** 2)
        else:
            solar_kw[i] = 0.0
    solar_series = pd.Series(solar_kw, index=pred_series.index, name="Solar_kW")

    # ---- Battery + grid simulation ----
    soc = 0.0  # state of charge in kWh
    grid_kw = np.zeros(len(pred_series), dtype=np.float32)
    battery_soc = np.zeros(len(pred_series), dtype=np.float32)

    for i, ts in enumerate(pred_series.index):
        demand = float(pred_series.iloc[i])  # use predicted demand for planning/simulation
        solar = float(solar_series.iloc[i])

        net = solar - demand  # positive => excess solar

        if net >= 0:
            # Charge battery with excess, limited by max charge rate and remaining capacity
            charge = min(net, cfg.battery_max_charge_kw, cfg.battery_capacity_kwh - soc)
            soc += charge
            grid = 0.0  # no grid needed when solar >= demand (excess may be curtailed)
        else:
            # Discharge battery to cover deficit, limited by max discharge rate and SOC
            deficit = -net
            discharge = min(deficit, cfg.battery_max_discharge_kw, soc)
            soc -= discharge
            grid = deficit - discharge

        grid_kw[i] = max(grid, 0.0)
        battery_soc[i] = soc

    grid_series = pd.Series(grid_kw, index=pred_series.index, name="Grid_kW")

    # ---- Three separate graphs (requested) ----
    # 1) Predicted demand vs time (include actual for context)
    plt.figure(figsize=(12, 4))
    plt.plot(pred_series.index, pred_series.values, label="Predicted", linewidth=1)
    plt.plot(actual_series.index, actual_series.values, label="Actual", linewidth=1, alpha=0.7)
    plt.title("Predicted Demand vs Time (Test Period)")
    plt.xlabel("Time")
    plt.ylabel("Demand (kW)")
    plt.legend()
    plt.tight_layout()

    # 2) Solar generation vs time
    plt.figure(figsize=(12, 3.5))
    plt.plot(solar_series.index, solar_series.values, label="Solar generation", linewidth=1, color="orange")
    plt.title("Solar Generation vs Time (Simulated)")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.tight_layout()

    # 3) Grid power needed vs time
    plt.figure(figsize=(12, 3.5))
    plt.plot(grid_series.index, grid_series.values, label="Grid power needed", linewidth=1, color="green")
    plt.title("Grid Power Needed vs Time (After Solar + Battery)")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.tight_layout()

    # Show all plots
    plt.show()


if __name__ == "__main__":
    # Make stdout flush nicely in some terminals
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()

