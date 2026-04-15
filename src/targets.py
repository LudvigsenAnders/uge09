
# src/targets.py
import pandas as pd
from config.runtime import CURRENT_RUN_MODE, RunMode


def _assert_training_mode():
    if CURRENT_RUN_MODE not in {RunMode.TRAIN, RunMode.BACKTEST}:
        raise RuntimeError(
            "add_targets_and_forward_returns() was called in LIVE mode.\n"
            "Targets are training-only and must never be used in live trading."
        )


def add_targets_and_forward_returns(
    df: pd.DataFrame,
    group_col: str,
    horizons=(1, 2, 3),
) -> pd.DataFrame:
    """
    TRAINING-ONLY FUNCTION.

    Adds forward returns and a binary target.
    Must NOT be called in live trading.
    """
    _assert_training_mode()

    for H in horizons:
        df[f"fwd_ret_{H}"] = (
            df.groupby(group_col)["Close"]
              .pct_change(H)
              .shift(-H)
        )

    df["future_close"] = (
        df.groupby(group_col)["Close"]
          .shift(-1)
    )

    df["target"] = (df["future_close"] > df["Close"]).astype(int)

    df = df.dropna(subset=["future_close"])
    df = df.drop(columns="future_close")

    return df
