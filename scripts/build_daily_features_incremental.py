
# scripts/build_daily_features_incremental.py

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
    from src.data_loader import load_prices, load_stock_list
    from src.feature_pipeline import FeaturePipeline

    # -------------------------------------------------
    # 3. Load configs
    # -------------------------------------------------
    STRATEGY_CONFIG = yaml.safe_load(open("config/strategy.yaml"))
    PATHS_CONFIG = yaml.safe_load(open("config/paths.yaml"))

    GROUP_COL = STRATEGY_CONFIG["group_col"]
    # BASE_COLS = STRATEGY_CONFIG["features"]["base_cols"]
    LAGS = STRATEGY_CONFIG["features"]["lags"]

    RAW_PRICES_PATH = Path(PATHS_CONFIG["raw_prices"])
    RAW_STOCK_LIST_PATH = Path(PATHS_CONFIG["raw_stock_list"])
    FEATURES_DIR = Path(PATHS_CONFIG["features_dir"])
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 4. Load raw data
    # -------------------------------------------------
    prices = load_prices(RAW_PRICES_PATH)
    stock_list = load_stock_list(RAW_STOCK_LIST_PATH)

    df = prices.merge(stock_list, on=GROUP_COL, how="left")

    # -------------------------------------------------
    # 5. Determine last processed date
    # -------------------------------------------------
    if any(FEATURES_DIR.iterdir()):
        existing_dates = [
            p.name.split("=")[1]
            for p in FEATURES_DIR.iterdir()
            if p.name.startswith("Date=")
        ]
        last_feature_date = pd.to_datetime(max(existing_dates))
    else:
        last_feature_date = None

    # -------------------------------------------------
    # 6. Restrict to new data + lookback
    # -------------------------------------------------
    # Lookback must be >= max(lags, rolling windows)
    LOOKBACK_DAYS = max(30, max(LAGS) + 10)

    if last_feature_date is not None:
        min_date = last_feature_date - pd.Timedelta(days=LOOKBACK_DAYS)
        df = df[df["Date"] > min_date]

    # -------------------------------------------------
    # 7. Core feature engineering (deterministic)
    # -------------------------------------------------
    df["return_1d"] = df.groupby(GROUP_COL)["Close"].pct_change()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Open"]
    df["vol_change"] = df.groupby(GROUP_COL)["Volume"].pct_change()

    df["vol_10"] = (
        df.groupby(GROUP_COL)["return_1d"]
        .rolling(10)
        .std()
        .shift(1)
        .reset_index(level=0, drop=True)
    )

    df["ret_vol_interaction"] = df["return_1d"] * df["vol_10"]

    df["vol_rank"] = (
        df.groupby("Date")["vol_10"]
        .rank(pct=True)
        .astype(float)
    )

    # -------------------------------------------------
    # 8. Lagged features (config-driven)
    # -------------------------------------------------
    pipe = FeaturePipeline.from_config(STRATEGY_CONFIG)
    df = pipe.add_grouped_lags(df)
    df, feature_cols = pipe.finalize_features(df)

    # -------------------------------------------------
    # 9. Keep only NEW dates
    # -------------------------------------------------
    if last_feature_date is not None:
        df = df[df["Date"] > last_feature_date]

    # -------------------------------------------------
    # 10. Persist features (partitioned Parquet)
    # -------------------------------------------------
    keep_cols = (
        ["Date", GROUP_COL, "Sector", "vol_rank"] + feature_cols
    )

    df_out = df[keep_cols].sort_values(["Date", GROUP_COL])
    df_out["Date"] = pd.to_datetime(df_out["Date"]).dt.date

    if not df_out.empty:
        df_out.to_parquet(
            FEATURES_DIR,
            partition_cols=["Date"],
            engine="pyarrow",
            index=False,
        )

        print(
            f"Features appended: "
            f"{df_out['Date'].min().date()} → {df_out['Date'].max().date()} "
            f"({len(df_out):,} rows)"
        )
    else:
        print("No new dates to process")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    main()
