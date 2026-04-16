
# src/backtest.py
import numpy as np
import pandas as pd
import copy

from src.signal_generation import generate_signals
from src.portfolio import compute_pnl_and_turnover


def walk_forward_backtest_slow(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    n_splits: int = 5,
    horizon: int = 3,
    cost: float = 0.001,
    quantile: float = 0.10,
    short_enabled: bool = False,
):
    results = []

    dates = df["Date"].sort_values().unique()
    splits = np.array_split(dates, n_splits + 1)

    X = df[feature_cols]
    y = df["target"]

    for i in range(n_splits):
        train_dates = np.concatenate(splits[: i + 1])
        test_dates = splits[i + 1]

        train_mask = df["Date"].isin(train_dates)
        test_mask = df["Date"].isin(test_dates)

        fold_model = copy.deepcopy(model)
        fold_model.fit(X.loc[train_mask], y.loc[train_mask])

        scores = fold_model.score(X.loc[test_mask])

        longs, shorts = generate_signals(
            scores=scores,
            sector=df.loc[scores.index, ["Sector"]]["Sector"],
            vol_rank=df.loc[scores.index, ["vol_rank"]]["vol_rank"],
            quantile=quantile,
            short_enabled=short_enabled,
        )

        test_ret = df.loc[scores.index, f"fwd_ret_{horizon}"]

        pnl_gross, pnl_net, turnover = compute_pnl_and_turnover(
            test_ret=test_ret,
            longs=longs,
            shorts=shorts,
            horizon=horizon,
            cost=cost,
            short_enabled=short_enabled,
        )

        results.append({
            "fold": i,
            "horizon": horizon,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "turnover": turnover,
            "short_enabled": short_enabled,
        })

    return pd.DataFrame(results)


def summarize(results):
    out = (
        results.groupby("horizon")[["pnl_net"]]
        .agg(["mean", "std"])
    )
    out["sharpe"] = (out[("pnl_net", "mean")] / out[("pnl_net", "std")])
    return out
