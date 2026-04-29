
# scripts/run_daily_signal.py


def main():
    # -------------------------------------------------
    # 1. Bootstrap & enforce LIVE mode
    # -------------------------------------------------
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    import config.runtime as runtime
    runtime.CURRENT_RUN_MODE = runtime.RunMode.LIVE

    # -------------------------------------------------
    # 2. Imports
    # -------------------------------------------------
    import yaml
    import pandas as pd
    import joblib
    # from src.data_loader import load_prices, load_stock_list
    from src.feature_pipeline import FeaturePipeline
    from src.signal_generation import generate_signals

    # -------------------------------------------------
    # 3. Load configs
    # -------------------------------------------------
    STRATEGY_CONFIG = yaml.safe_load(open("config/strategy.yaml"))
    PATHS_CONFIG = yaml.safe_load(open("config/paths.yaml"))

    GROUP_COL = STRATEGY_CONFIG["group_col"]
    # BASE_COLS = STRATEGY_CONFIG["features"]["base_cols"]
    # LAGS = STRATEGY_CONFIG["features"]["lags"]

    QUANTILE = STRATEGY_CONFIG["portfolio"]["quantile"]
    SHORT_ENABLED = STRATEGY_CONFIG["portfolio"]["short_enabled"]

    FEATURES_DIR = Path(PATHS_CONFIG["features_dir"])
    MODEL_PATH = Path(PATHS_CONFIG["model_path"])
    SIGNALS_DIR = Path(PATHS_CONFIG["signals_dir"])
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 4. Load precomputed features (partitioned Parquet)
    # -------------------------------------------------
    df_features = pd.read_parquet(FEATURES_DIR)
    print(df_features.info())
    if not pd.api.types.is_datetime64_any_dtype(df_features["Date"]):
        df_features["Date"] = pd.to_datetime(df_features["Date"]).dt.date

    today = df_features["Date"].max()
    df_today = df_features[df_features["Date"] == today].copy()

    assert not df_today.empty, "No features found for latest date"

    # -------------------------------------------------
    # 5. Feature pipeline (normalization ONLY)
    # -------------------------------------------------

    pipe = FeaturePipeline.from_config(STRATEGY_CONFIG)
    df_today = pipe.cross_sectional_normalize(df_today)

    # -------------------------------------------------
    # 6. Load trained model
    # -------------------------------------------------
    model = joblib.load(MODEL_PATH)

    scores = model.score(df_today[pipe.feature_cols_])

    # -------------------------------------------------
    # 7. Generate signals (pure logic)
    # -------------------------------------------------
    longs, shorts = generate_signals(
        scores=scores,
        sector=df_today["Sector"],
        vol_rank=df_today["vol_rank"],
        quantile=QUANTILE,
        short_enabled=SHORT_ENABLED,
    )

    # -------------------------------------------------
    # 8. Build signal output
    # -------------------------------------------------
    signals = pd.DataFrame({
        "Date": today,
        GROUP_COL: df_today[GROUP_COL],
        "score": scores,
        "is_long": longs.astype(int),
    })

    if SHORT_ENABLED:
        signals["is_short"] = shorts.astype(int)

    # Keep only actionable rows
    signals = signals[
        (signals["is_long"] == 1) | (signals.get("is_short", 0) == 1)
    ]

    OUT_PATH = SIGNALS_DIR / f"{today}.csv"
    signals.to_csv(OUT_PATH, index=False)

    print(
        f"Daily signals written: {OUT_PATH}\n"
        f"Run mode: {runtime.CURRENT_RUN_MODE}\n"
        f"Short enabled: {SHORT_ENABLED}\n"
        f"Rows: {len(signals)}"
    )


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main()
