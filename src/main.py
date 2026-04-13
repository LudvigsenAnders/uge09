import pandas as pd
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
from datetime import datetime


db_path = Path("data/stock_OHLC.sqlite")
conn = sqlite3.connect(db_path)

# tickers = ["ticker_1", "ticker_2", "ticker_3", "ticker_4", "ticker_5", "ticker_6", "ticker_7", "ticker_8", "ticker_9", "ticker_10"]
# placeholders = ",".join("?" for _ in tickers)
# sql = f"""
# SELECT *
# FROM OHLC_train
# WHERE Ticker IN ({placeholders})
# """
# df = pd.read_sql(sql, conn, params=tickers)

sql = f"""
SELECT *
FROM OHLC_train"""

df = pd.read_sql(sql, conn)
conn.close()



print(df.info())

df["Date"] = pd.to_datetime(df["Date"])
print(df.info())

horizon = 3  # peridiction horizon: H=1 => t+1

start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 1, 1)
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

print(df.info())

##### Feature engineering
df = df.sort_values(by=["Ticker", "Date"])

df['target'] = 0
df.loc[df.groupby('Ticker')['Close'].diff(1) > 0, 'target'] = 1  # target is above yesterday
df.loc[df.groupby('Ticker')['Close'].diff(1) <= 0, 'target'] = 0  # target is below yesterday


df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek
df["days_in_month"] = df["Date"].dt.days_in_month
df["day"] = df["Date"].dt.day

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["is_start_of_month"] = (df["day"] <= 3).astype(int)
df["is_end_of_month"] = (df["days_in_month"] - df["day"] < 3).astype(int)


lag_cols = ["Open", "High", "Low", "Close", "Volume"]
lags = [3, 5, 10]
mean_cols = ["Open", "High", "Low", "Close", "Volume"]
std_cols = ["Open", "High", "Low", "Close"]
windows = [5]

for lag in lags:
    df[[f"{c}_lag_{lag}" for c in lag_cols]] = (
        df.groupby("Ticker")[lag_cols]
          .shift(lag)
    )

for col in mean_cols:
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = (
            df.groupby("Ticker")[col]
              .shift(horizon)
              .rolling(w)
              .mean()
        )

for col in std_cols:
    for w in windows:
        df[f"{col}_roll_std_{w}"] = (
            df.groupby("Ticker")[col]
            .shift(horizon)
            .rolling(w)
            .std()
        )


import pandas as pd
from typing import Iterable


def add_return_volatility(
    df: pd.DataFrame,
    price_col: str,
    windows: Iterable[int],
    horizon: int,
    *,
    group_col: str | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Add rolling volatility of returns aligned to prediction horizon
    for multiple window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    price_col : str
        Column containing prices (e.g. 'Close')
    windows : Iterable[int]
        Rolling window sizes (e.g. [5, 10, 20])
    horizon : int
        Prediction horizon H (e.g. 5)
    group_col : str | None
        Column to group by (e.g. 'ticker'); None for single series
    prefix : str | None
        Optional prefix for generated column names

    Returns
    -------
    pd.DataFrame
        DataFrame with added volatility columns
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if prefix is None:
        prefix = f"{price_col}_ret_vol"

    # Ensure sorted order to avoid silent leakage
    if group_col:
        df = df.sort_values([group_col])
    else:
        df = df.sort_index()

    # 1. Compute 1‑period returns
    if group_col:
        returns = (
            df.groupby(group_col)[price_col]
              .pct_change()
        )
    else:
        returns = df[price_col].pct_change()

    # 2. Rolling volatility for each window
    for w in windows:
        if w <= 1:
            raise ValueError("rolling window must be > 1")

        col_name = f"{prefix}_{w}"

        if group_col:
            df[col_name] = (
                returns
                .groupby(df[group_col])
                .shift(horizon)
                .rolling(w)
                .std()
            )
        else:
            df[col_name] = (
                returns
                .shift(horizon)
                .rolling(w)
                .std()
            )

    return df


df = add_return_volatility(
    df,
    price_col="Close",
    windows=[10],
    horizon=horizon,
    group_col="Ticker"
)



df = df.dropna().reset_index(drop=True)

#print(df.info())
#print(df.head(30))


def build_lag_col_names(base_cols, lags):
    return [
        f"{c}_lag_{l}"
        for c in base_cols
        for l in lags
    ]


def build_roll_col_names(base_cols, roll_windows, roll_stats=("mean", "std")):
    return [
        f"{c}_roll_{s}_{w}"
        for c in base_cols
        for s in roll_stats
        for w in roll_windows
    ]




lag_col_names = build_lag_col_names(base_cols=lag_cols, lags=lags)
mean_col_names = build_roll_col_names(base_cols=mean_cols, roll_windows=windows, roll_stats=["mean"])
std_col_names = build_roll_col_names(base_cols=std_cols, roll_windows=windows, roll_stats=["std"])

ret_name = ["Close_ret_vol_10"]

num_cols = lag_col_names + ret_name #+ mean_col_names + std_col_names

print("Numeric feature columns:", num_cols)
# num_cols = [
#     "Open_lag_1",
#     "High_lag_1",
#     "Low_lag_1",
#     "Close_lag_1",
#     "Volume_lag_1",
#     "Open_lag_3",
#     "High_lag_3",
#     "Low_lag_3",
#     "Close_lag_3",
#     "Volume_lag_3",
#     "Open_lag_7",
#     "High_lag_7",
#     "Low_lag_7",
#     "Close_lag_7",
#     "Volume_lag_7",
#     "Open_roll_mean_7",
#     "High_roll_mean_7",
#     "Low_roll_mean_7",
#     "Close_roll_mean_7",
#     "Volume_roll_mean_7",
#     # "Open_roll_mean_14",
#     # "High_roll_mean_14",
#     # "Low_roll_mean_14",
#     # "Close_roll_mean_14",
#     # "Volume_roll_mean_14",
#     "Open_roll_std_7",
#     "High_roll_std_7",
#     "Low_roll_std_7",
#     "Close_roll_std_7",
#     # "Open_roll_std_14",
#     # "High_roll_std_14",
#     # "Low_roll_std_14",
#     # "Close_roll_std_14",
#     "month_sin",
#     "month_cos",
#     "dow_sin",
#     "dow_cos",
#     "is_start_of_month",
#     "is_end_of_month"
# ]

cat_cols = [
    "Ticker"  # optional, see previous guidance
]
df[cat_cols] = df[cat_cols].astype("category")

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)


X = df[num_cols + cat_cols]
y = df["target"]

#split_date = "2024-07-01"

#train = df[df["timestamp"] < split_date]
#test  = df[df["timestamp"] >= split_date]

# X_train = train[num_cols + cat_cols]
# y_train = train["target"]

# X_test  = test[num_cols + cat_cols]
# y_test  = test["target"]

log_reg = Pipeline([
    ("preprocess", preprocess),
    ("log_reg", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    ))
])



from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

xgb_model = Pipeline([
    ("preprocess", preprocess),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",      # REQUIRED for speed + categorical support
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])




scores = []


train_size = 120      # fixed training window
val_size = 30         # fixed validation horizon
step = 30             # how far the window moves

for start in range(0, len(X) - train_size - val_size + 1, step):
    train_idx = slice(start, start + train_size)
    val_idx = slice(start + train_size, start + train_size + val_size)

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    log_reg.fit(X_tr, y_tr)
    y_val_proba = log_reg.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, y_val_proba)
    scores.append(auc)

print("\nstd_dev AUC:", np.std(scores))
print("\nMean AUC:", np.mean(scores))

#print(log_reg.named_steps["log_reg"].feature_importances_)
