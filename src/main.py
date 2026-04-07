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
df['trend'] = 0
df.loc[df.groupby('Ticker')['Close'].diff() > 0, 'trend'] = 1
df.loc[df.groupby('Ticker')['Close'].diff() <= 0, 'trend'] = 0




#df_ticker_1 = df[df["Ticker"] == "ticker_1"]
df_ticker_y2020 = df[df["Date"].dt.year == 2020]

fig1 = df_ticker_y2020.plot(x="Date", y="Close", kind="line", title="Closing Price Over Time")

print(df_ticker_y2020.head(10))
print(df_ticker_y2020.tail(10))

plt.show()


