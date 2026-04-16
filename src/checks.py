
# src/checks.py

import pandas as pd
from src.diagnostics import max_drawdown


def run_backtest_checks(
    results: pd.DataFrame,
    equity: pd.Series,
    max_allowed_drawdown: float = -0.20,
):
    """
    Fail-fast regression checks for backtests.
    """

    # --- basic sanity ---
    assert results["pnl_net"].notna().all(), "NaNs detected in pnl_net"
    assert results["turnover"].mean() < 0.10, "Turnover unexpectedly high"

    sharpe = results["pnl_net"].mean() / results["pnl_net"].std()
    assert sharpe > 0.0, f"Sharpe <= 0 ({sharpe:.2f})"

    # --- drawdown constraint ---
    mdd = max_drawdown(equity)
    assert mdd >= max_allowed_drawdown, (
        f"Max drawdown too large: {mdd:.2%} "
        f"(limit {max_allowed_drawdown:.0%})"
    )

    return {
        "sharpe": sharpe,
        "mean_pnl": results["pnl_net"].mean(),
        "turnover": results["turnover"].mean(),
        "max_drawdown": mdd,
    }
