
from typing import List, Iterable, Tuple
import pandas as pd
import numpy as np
import copy
from pathlib import Path
import sqlite3
from sklearn.linear_model import LogisticRegression


class FeaturePipeline:
    def __init__(
        self,
        group_col: str,
        base_cols: List[str],
        lags: Iterable[int],
    ):
        self.group_col = group_col
        self.base_cols = base_cols
        self.lags = list(lags)
        self.feature_cols_: List[str] = []

    def add_grouped_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.base_cols:
            for lag in self.lags:
                df[f"{col}_lag{lag}"] = (
                    df.groupby(self.group_col)[col].shift(lag)
                )
        return df

    def finalize_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        feature_cols = (self.base_cols + [f"{col}_lag{lag}" for col in self.base_cols for lag in self.lags])

        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        self.feature_cols_ = feature_cols
        return df, feature_cols

    def cross_sectional_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.feature_cols_] = (
            df.groupby("Date")[self.feature_cols_]
              .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
        )
        return df


class AlphaModel:
    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        score = self.model.predict_proba(X)[:, 1]
        return pd.Series(score, index=X.index)


def add_targets_and_forward_returns(
    df: pd.DataFrame,
    group_col: str,
    horizons=(1, 2, 3),
):
    for H in horizons:
        df[f"fwd_ret_{H}"] = (
            df.groupby(group_col)["Close"]
              .pct_change(H)
              .shift(-H)
        )

    df["future_close"] = df.groupby(group_col)["Close"].shift(-1)
    df["target"] = (df["future_close"] > df["Close"]).astype(int)
    df = df.dropna(subset=["future_close"])
    df = df.drop(columns="future_close")

    return df


def walk_forward_backtest_slow(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: AlphaModel,
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

        # Eligibility filter
        eligible = (
            df.loc[scores.index, "vol_rank"]
            .astype(float) < 0.7
        )

        # Sector-neutral ranking
        sector = df.loc[scores.index, "Sector"]
        ranks = scores.where(eligible).groupby(sector).rank(pct=True)

        longs = ranks > (1.0 - quantile)

        
        if short_enabled:
            shorts = ranks < quantile
        else:
            shorts = pd.Series(False, index=ranks.index)

        test_ret = df.loc[scores.index, f"fwd_ret_{horizon}"]

        if short_enabled:
            if longs.sum() == 0 or shorts.sum() == 0:
                pnl_gross = 0.0
                turnover = 0.0
            else:
                pnl_gross = (
                    test_ret[longs].mean()
                    - test_ret[shorts].mean()
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

##################
##################


db_path = Path("data/jpx/JPX_stocks.sqlite")
conn = sqlite3.connect(db_path)

df = pd.read_sql("SELECT * FROM stock_prices", conn)
df_stock_list = pd.read_sql("SELECT * FROM stock_list", conn)
conn.close()

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

# --- Core slow features ---
df["return_1d"] = df.groupby("SecuritiesCode")["Close"].pct_change()
df["hl_range"] = (df["High"] - df["Low"]) / df["Open"]
df["vol_change"] = df.groupby("SecuritiesCode")["Volume"].pct_change()

df["vol_10"] = (
    df.groupby("SecuritiesCode")["return_1d"]
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

# Sector (explicit unknown category)
df_stock_list["Sector"] = (
    df_stock_list["17SectorCode"]
    .astype("Int64")
    .fillna(-1)
)

df = df.merge(
    df_stock_list[["SecuritiesCode", "Sector"]],
    on="SecuritiesCode",
    how="left",
)

group_col = "SecuritiesCode"
base_cols = [
    "return_1d",
    "hl_range",
    "vol_change",
    "vol_10",
    "ret_vol_interaction",
]
lags = [1, 2, 3]

pipe = FeaturePipeline(group_col, base_cols, lags)
df = pipe.add_grouped_lags(df)
df, feature_cols = pipe.finalize_features(df)

df = add_targets_and_forward_returns(df, group_col)
df = df.dropna(
    subset=feature_cols + ["target", "fwd_ret_3"]
).reset_index(drop=True)

df = pipe.cross_sectional_normalize(df)

assert np.isfinite(df[feature_cols].values).all()

slow_alpha = AlphaModel(
    LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    ),
    name="SlowMomentum",
)

results_long_only = walk_forward_backtest_slow(
    df=df,
    feature_cols=feature_cols,
    model=slow_alpha,
    horizon=3,
    cost=0.001,
    short_enabled=False,
)

print(summarize(results_long_only))
print(results_long_only)


results_long_short = walk_forward_backtest_slow(
    df=df,
    feature_cols=feature_cols,
    model=slow_alpha,
    horizon=3,
    cost=0.001,
    short_enabled=True,
)

print(summarize(results_long_short))
print(results_long_short)