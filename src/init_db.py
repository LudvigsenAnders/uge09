import pandas as pd
from pathlib import Path
import sqlite3

csv_train_path = Path("data/train.csv")
csv_test_path = Path("data/test.csv")
db_path = Path("data/stock_OHLC.sqlite")


# Read CSV
df_train = pd.read_csv(csv_train_path)
df_test = pd.read_csv(csv_test_path)

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Write to SQLite table
df_train.to_sql(
    name="OHLC_train",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_test.to_sql(
    name="OHLC_test",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

# Close connection
conn.close()
