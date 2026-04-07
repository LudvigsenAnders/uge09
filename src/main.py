import pandas as pd
from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier




db_path = Path("data/stock_OHLC.sqlite")
conn = sqlite3.connect(db_path)

tickers = ["ticker_1", "ticker_2", "ticker_3", "ticker_4", "ticker_5", "ticker_6", "ticker_7", "ticker_8", "ticker_9", "ticker_10"]

placeholders = ",".join("?" for _ in tickers)


sql = f"""
SELECT *
FROM OHLC_train
WHERE Ticker IN ({placeholders})
"""

df = pd.read_sql(sql, conn, params=tickers)


conn.close()

print(df.info())

df["Date"] = pd.to_datetime(df["Date"])
print(df.info())

##### Feature engineering
df = df.sort_values(by=["Ticker", "Date"])
df['target'] = 0
df.loc[df.groupby('Ticker')['Close'].diff() > 0, 'target'] = 1
df.loc[df.groupby('Ticker')['Close'].diff() <= 0, 'target'] = 0


cat_cols = ["Ticker"]

df[cat_cols] = df[cat_cols].astype("category")


df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day_of_month"] = df["Date"].dt.day
df = df.drop(columns=["Date"])

print(df.info())


X = df.drop(columns=["target"])
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    enable_categorical=True,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42

)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)



print("ROC‑AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# #df_ticker_1 = df[df["Ticker"] == "ticker_1"]
# df_ticker_y2020 = df[df["Date"].dt.year == 2020]

# fig1 = df_ticker_y2020.plot(x="Date", y="Close", kind="line", title="Closing Price Over Time")

# print(df_ticker_y2020.head(10))
# print(df_ticker_y2020.tail(10))

# plt.show()


