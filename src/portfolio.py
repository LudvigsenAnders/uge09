
# src/portfolio.py
import pandas as pd


def compute_pnl_and_turnover(
    test_ret: pd.Series,
    longs: pd.Series,
    shorts: pd.Series,
    horizon: int,
    cost: float,
    short_enabled: bool,
):
    if short_enabled:
        if longs.sum() == 0 or shorts.sum() == 0:
            pnl_gross = 0.0
            turnover = 0.0
        else:
            pnl_gross = (
                test_ret[longs].mean() - test_ret[shorts].mean()
            )
            turnover = (longs.mean() + shorts.mean()) / horizon
    else:
        if longs.sum() == 0:
            pnl_gross = 0.0
            turnover = 0.0
        else:
            pnl_gross = test_ret[longs].mean()
            turnover = longs.mean() / horizon

    pnl_net = pnl_gross - cost * turnover
    return pnl_gross, pnl_net, turnover
