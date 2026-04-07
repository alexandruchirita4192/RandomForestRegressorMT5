from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None


FEATURE_COLS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_10",
    "vol_20",
    "dist_sma_10",
    "dist_sma_20",
    "zscore_20",
    "atr_14",
]


def fetch_rates_from_mt5(symbol: str, timeframe_name: str, bars: int) -> pd.DataFrame:
    if mt5 is None:
        raise RuntimeError(
            "Pachetul MetaTrader5 pentru Python nu este instalat. Instaleaza-l cu: pip install MetaTrader5"
        )

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe_name not in timeframe_map:
        raise ValueError(f"Timeframe nesuportat: {timeframe_name}")

    if not mt5.initialize():
        raise RuntimeError(f"initialize() a esuat: {mt5.last_error()}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe_name], 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Nu am putut citi datele pentru {symbol} {timeframe_name}. last_error={mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]].copy()
    finally:
        mt5.shutdown()



def load_rates_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {"time", "open", "high", "low", "close"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV-ul nu contine coloanele obligatorii: {sorted(missing)}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    return df[["time", "open", "high", "low", "close", "volume"]]



def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)

    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["dist_sma_10"] = (df["close"] / df["sma_10"]) - 1.0
    df["dist_sma_20"] = (df["close"] / df["sma_20"]) - 1.0

    roll_mean_20 = df["close"].rolling(20).mean()
    roll_std_20 = df["close"].rolling(20).std()
    df["zscore_20"] = (df["close"] - roll_mean_20) / roll_std_20

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = df["tr"].rolling(14).mean()

    # Target: randamentul urmatorului bar, calculat doar din bare inchise.
    df["fwd_ret_1"] = df["close"].shift(-1) / df["close"] - 1.0

    df = df.dropna(subset=FEATURE_COLS + ["fwd_ret_1"]).copy()
    return df



def walk_forward_report(df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS].astype(np.float32)
    y = df["fwd_ret_1"].astype(np.float32)

    tscv = TimeSeriesSplit(n_splits=5)
    rmses: List[float] = []
    maes: List[float] = []
    directional_accs: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        direction = float((np.sign(pred) == np.sign(y_test)).mean())

        rmses.append(rmse)
        maes.append(mae)
        directional_accs.append(direction)
        print(f"Fold {fold}: RMSE={rmse:.8f} MAE={mae:.8f} DirectionalAcc={direction:.4f}")

    return {
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "directional_accuracy_mean": float(np.mean(directional_accs)),
    }



def fit_final_model(df: pd.DataFrame) -> RandomForestRegressor:
    X = df[FEATURE_COLS].astype(np.float32)
    y = df["fwd_ret_1"].astype(np.float32)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model



def export_to_onnx(model: RandomForestRegressor, output_path: Path) -> None:
    initial_type = [("float_input", FloatTensorType([1, len(FEATURE_COLS)]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=15)
    output_path.write_bytes(onx.SerializeToString())



def compute_suggested_threshold(df: pd.DataFrame, quantile: float = 0.60) -> float:
    threshold = float(df["fwd_ret_1"].abs().quantile(quantile))
    return max(threshold, 1e-6)



def save_metadata(output_dir: Path, report: dict, threshold: float, symbol: str, timeframe: str) -> None:
    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "features": FEATURE_COLS,
        "entry_threshold": threshold,
        "model_type": "RandomForestRegressor",
        "training_notes": {
            "target": "next closed-bar return",
            "signal": "buy if prediction > +threshold, sell if prediction < -threshold, else flat",
        },
        "walk_forward": report,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Antreneaza un model ML pentru MT5 si exporta ONNX.")
    p.add_argument("--symbol", default="EURUSD", help="Simbolul folosit la training")
    p.add_argument("--timeframe", default="H1", help="M1/M5/M15/M30/H1/H4/D1")
    p.add_argument("--bars", type=int, default=15000, help="Numar de bare de citit din MT5")
    p.add_argument("--csv", type=str, default="", help="Alternativ, citeste datele din CSV")
    p.add_argument("--output-dir", default="output", help="Directorul de output")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        raw = load_rates_from_csv(Path(args.csv))
    else:
        raw = fetch_rates_from_mt5(args.symbol, args.timeframe, args.bars)

    raw.to_csv(output_dir / "training_rates_snapshot.csv", index=False)
    feat_df = build_features(raw)
    feat_df.to_csv(output_dir / "training_features_snapshot.csv", index=False)

    print(f"Set de antrenare: {len(feat_df)} randuri")
    report = walk_forward_report(feat_df)
    print("\nRezumat walk-forward:")
    print(json.dumps(report, indent=2))

    model = fit_final_model(feat_df)
    onnx_path = output_dir / "ml_strategy_model.onnx"
    export_to_onnx(model, onnx_path)

    threshold = compute_suggested_threshold(feat_df, quantile=0.60)
    save_metadata(output_dir, report, threshold, args.symbol, args.timeframe)

    print(f"\nModel ONNX salvat in: {onnx_path}")
    print(f"Prag initial sugerat pentru EA: {threshold:.8f}")
    print("Copiaza fisierul ml_strategy_model.onnx langa EA-ul .mq5 inainte de compilare.")


if __name__ == "__main__":
    main()
