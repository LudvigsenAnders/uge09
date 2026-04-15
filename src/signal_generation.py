# src/signal_generation.py
import pandas as pd


def generate_signals(
    scores: pd.Series,
    sector: pd.Series,
    vol_rank: pd.Series,
    quantile: float,
    short_enabled: bool,
):
    eligible = vol_rank.astype(float) < 0.7

    ranks = scores.where(eligible).groupby(sector).rank(pct=True)

    longs = ranks > (1.0 - quantile)

    if short_enabled:
        shorts = ranks < quantile
    else:
        shorts = pd.Series(False, index=ranks.index)

    return longs, shorts
