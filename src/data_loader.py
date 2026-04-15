# src/data_loader.py
import pandas as pd
from pathlib import Path
import sqlite3


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)


def load_stock_list(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["Sector"] = (
        df["17SectorCode"]
        .astype("Int64")
        .fillna(-1)
    )
    return df[["SecuritiesCode", "Sector"]]


def load_from_sqlite():
    db_path = Path("data/jpx/JPX_stocks.sqlite")
    conn = sqlite3.connect(db_path)

    df = pd.read_sql("SELECT * FROM stock_prices", conn)
    df_stock_list = pd.read_sql("SELECT * FROM stock_list", conn)
    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)

    df.to_parquet("data/raw/stock_prices/stock_prices.parquet", index=False)
    df_stock_list.to_parquet("data/raw/stock_list/stock_list.parquet", index=False)
