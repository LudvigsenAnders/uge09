
# src/diagnostics.py

import pandas as pd
import matplotlib.pyplot as plt

from src.signal_generation import generate_signals
from src.portfolio import compute_pnl_and_turnover


def compute_daily_pnl_series(
    df: pd.DataFrame,
    feature_cols: list[str],
    model,
    horizon: int,
    quantile: float,
    short_enabled: bool,
) -> pd.DataFrame:
    """
    Compute an expanding-window daily PnL series.

    Diagnostic use only:
    - no transaction costs
    - no fold aggregation
    - preserves time ordering
    """

    dates = sorted(df["Date"].unique())
    pnl_records = []

    # We need enough history for training and forward returns
    for t in range(horizon, len(dates) - horizon):
        train_dates = dates[:t]
        test_date = dates[t]

        train_mask = df["Date"].isin(train_dates)
        test_mask = df["Date"] == test_date

        # Fresh model instance (no leakage)
        base_model = model.model.__class__(**model.model.get_params())
        fold_model = model.__class__(base_model)

        fold_model.fit(
            df.loc[train_mask, feature_cols],
            df.loc[train_mask, "target"],
        )

        scores = fold_model.score(df.loc[test_mask, feature_cols])

        longs, shorts = generate_signals(
            scores=scores,
            sector=df.loc[scores.index, ["Sector"]]["Sector"],
            vol_rank=df.loc[scores.index, ["vol_rank"]]["vol_rank"],
            quantile=quantile,
            short_enabled=short_enabled,
        )

        test_ret = df.loc[scores.index, f"fwd_ret_{horizon}"]

        _, pnl_net, _ = compute_pnl_and_turnover(
            test_ret=test_ret,
            longs=longs,
            shorts=shorts,
            horizon=horizon,
            cost=0.0,          # diagnostics: no costs
            short_enabled=short_enabled,
        )

        pnl_records.append({
            "Date": test_date,
            "pnl": pnl_net,
        })

    return (
        pd.DataFrame(pnl_records)
        .set_index("Date")
        .sort_index()
    )


def plot_equity_curve(
    pnl: pd.Series | pd.DataFrame,
    title: str = "Equity Curve",
):
    """
    Plot equity curve from daily PnL.

    Accepts:
    - Series of daily PnL
    - DataFrame with column 'pnl'
    """
    if isinstance(pnl, pd.DataFrame):
        if "pnl" not in pnl.columns:
            raise ValueError("DataFrame must contain a 'pnl' column")
        pnl = pnl["pnl"]

    equity = (1.0 + pnl).cumprod()

    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values, label="Strategy")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    print("\n NEXT UP PLOT SHOW")

    plt.show()

    return equity


def summarize_backtest(results):
    out = (
        results.groupby("horizon")[["pnl_net"]]
        .agg(["mean", "std"])
    )
    out["sharpe"] = (out[("pnl_net", "mean")] / out[("pnl_net", "std")])
    return out



def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from an equity curve.

    Drawdown_t = (Equity_t - max(Equity_{<=t})) / max(Equity_{<=t})
    """
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown (most negative drawdown).
    """
    return compute_drawdown(equity).min()

def drawdown_duration(equity: pd.Series) -> pd.Series:
    """
    Drawdown duration series (time under water).

    Counts number of consecutive periods since last equity peak.
    Resets to 0 at new highs.
    """
    running_max = equity.cummax()
    underwater = equity < running_max

    duration = []
    current = 0

    for is_underwater in underwater:
        if is_underwater:
            current += 1
        else:
            current = 0
        duration.append(current)

    return pd.Series(duration, index=equity.index)


def max_drawdown_duration(equity: pd.Series) -> int:
    """
    Maximum drawdown duration (longest time under water).
    """
    return drawdown_duration(equity).max()


def plot_drawdown(equity: pd.Series, title: str = "Drawdown"):
    """
    Plot drawdown curve.
    """
    dd = compute_drawdown(equity)

    plt.figure(figsize=(10, 3))
    plt.fill_between(dd.index, dd.values, 0.0, color="red", alpha=0.4)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return dd


def plot_drawdown_duration(
    equity: pd.Series,
    title: str = "Drawdown Duration",
):
    """
    Plot drawdown duration (time under water).
    """
    duration = drawdown_duration(equity)

    plt.figure(figsize=(10, 3))
    plt.plot(duration.index, duration.values, color="orange")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Days Under Water")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return duration


# -------------------------------------------------
# Manual test
# -------------------------------------------------
if __name__ == "__main__":
    example_equity = pd.Series(
        [1.0, 1.1, 1.05, 1.02, 1.15, 1.14, 1.20],
        index=pd.date_range("2021-01-01", periods=7),
    )

    print("Max drawdown:", max_drawdown(example_equity))
    print("Max drawdown duration:", max_drawdown_duration(example_equity))

    plot_drawdown(example_equity)
    plot_drawdown_duration(example_equity)
