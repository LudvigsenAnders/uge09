
# scripts/run_backtest.py

# -------------------------------------------------
# 1. Bootstrap & enforce BACKTEST mode
# -------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config.runtime as runtime
runtime.CURRENT_RUN_MODE = runtime.RunMode.BACKTEST


# -------------------------------------------------
# 2. Imports
# -------------------------------------------------
import yaml
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_prices, load_stock_list
from src.feature_pipeline import FeaturePipeline
from src.model import AlphaModel
from src.targets import add_targets_and_forward_returns
from src.backtest import walk_forward_backtest_slow


# -------------------------------------------------
# 3. Load configs
# -------------------------------------------------
STRATEGY_CONFIG = yaml.safe_load(open("config/strategy.yaml"))
PATHS_CONFIG = yaml.safe_load(open("config/paths.yaml"))

GROUP_COL = STRATEGY_CONFIG["group_col"]

RAW_PRICES_PATH = Path(PATHS_CONFIG["raw_prices"])
RAW_STOCK_LIST_PATH = Path(PATHS_CONFIG["raw_stock_list"])

QUANTILE = STRATEGY_CONFIG["portfolio"]["quantile"]
SHORT_ENABLED = STRATEGY_CONFIG["portfolio"]["short_enabled"]
COST = STRATEGY_CONFIG["risk"]["cost"]


# -------------------------------------------------
# 4. Load raw data
# -------------------------------------------------
prices = load_prices(RAW_PRICES_PATH)
stock_list = load_stock_list(RAW_STOCK_LIST_PATH)

df = prices.merge(stock_list, on=GROUP_COL, how="left")
df["Date"] = pd.to_datetime(df["Date"])


# -------------------------------------------------
# 5. Feature engineering (panel)
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
# 6. Features (config-driven)
# -------------------------------------------------
pipe = FeaturePipeline.from_config(STRATEGY_CONFIG)

df = pipe.add_grouped_lags(df)
df, feature_cols = pipe.finalize_features(df)


# -------------------------------------------------
# 7. Targets (BACKTEST ONLY)
# -------------------------------------------------
df = add_targets_and_forward_returns(df, GROUP_COL)

df = df.dropna(subset=["target"]).reset_index(drop=True)


# -------------------------------------------------
# 8. Cross-sectional normalization
# -------------------------------------------------
df = pipe.cross_sectional_normalize(df)


# -------------------------------------------------
# 9. Model definition
# -------------------------------------------------
model = AlphaModel(
    LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    ),
    name="SlowMomentum",
)


# -------------------------------------------------
# 10. Walk-forward backtest
# -------------------------------------------------
results = walk_forward_backtest_slow(
    df=df,
    feature_cols=feature_cols,
    model=model,
    horizon=STRATEGY_CONFIG["portfolio"]["holding_period"],
    cost=COST,
    quantile=QUANTILE,
    short_enabled=SHORT_ENABLED,
)


# -------------------------------------------------
# 11. Summary
# -------------------------------------------------
summary = (
    results.groupby("horizon")[["pnl_net"]]
    .agg(["mean", "std"])
)

summary["sharpe"] = (
    summary[("pnl_net", "mean")] /
    summary[("pnl_net", "std")]
)

print("\n=== BACKTEST SUMMARY ===")
print(summary)

print("\n=== FOLD DETAILS ===")
print(results)

print(f"\nRun mode: {runtime.CURRENT_RUN_MODE}")
print(f"Rows used: {len(df):,}")
print(f"Features: {len(feature_cols)}")
