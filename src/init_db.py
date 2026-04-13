from typing import Mapping, Hashable
from pandas._typing import DtypeArg
import pandas as pd
from pathlib import Path
import sqlite3

csv_financials_path = Path("data/jpx/train_files/financials.csv")
csv_options_path = Path("data/jpx/train_files/options.csv")
csv_secondary_stock_prices_path = Path("data/jpx/train_files/secondary_stock_prices.csv")
csv_stock_prices_path = Path("data/jpx/train_files/stock_prices.csv")
csv_trades_path = Path("data/jpx/train_files/trades.csv")
csv_stock_list_path = Path("data/jpx/stock_list.csv")
db_path = Path("data/jpx/JPX_stocks.sqlite")

stock_prices_dtypes: Mapping[Hashable, DtypeArg] = {
    "RowId": "string",
    "Date": "string",

    "SecuritiesCode": "Int64",

    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",

    "Volume": "Int64",
    "AdjustmentFactor": "float64",

    "ExpectedDividend": "float64",
    "SupervisionFlag": "boolean",
}

secondary_stock_prices_dtypes: Mapping[Hashable, DtypeArg] = {
    "RowId": "string",
    "Date": "string",

    "SecuritiesCode": "Int64",

    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",

    "Volume": "Int64",
    "AdjustmentFactor": "float64",

    "ExpectedDividend": "float64",
    "SupervisionFlag": "boolean",
}

options_dtypes: Mapping[Hashable, DtypeArg] = {
    "DateCode": "string",
    "OptionsCode": "string",

    "WholeDayOpen": "float64",
    "WholeDayHigh": "float64",
    "WholeDayLow": "float64",
    "WholeDayClose": "float64",

    "NightSessionOpen": "float64",
    "NightSessionHigh": "float64",
    "NightSessionLow": "float64",
    "NightSessionClose": "float64",

    "DaySessionOpen": "float64",
    "DaySessionHigh": "float64",
    "DaySessionLow": "float64",
    "DaySessionClose": "float64",

    "TradingVolume": "Int64",
    "OpenInterest": "Int64",
    "TradingValue": "Int64",

    "ContractMonth": "Int64",
    "StrikePrice": "float64",
    "WholeDayVolume": "Int64",
    "Putcall": "Int64",

    "SettlementPrice": "float64",
    "TheoreticalPrice": "float64",
    "BaseVolatility": "float64",
    "ImpliedVolatility": "float64",
    "InterestRate": "float64",
    "DividendRate": "float64",
    "Dividend": "float64",
}

trades_dtypes: Mapping[Hashable, DtypeArg] = {
    "Date": "string",
    "StartDate": "string",
    "EndDate": "string",
    "Section": "string",

    "TotalSales": "float64",
    "TotalPurchases": "float64",
    "TotalTotal": "float64",
    "TotalBalance": "float64",

    "ProprietarySales": "float64",
    "ProprietaryPurchases": "float64",
    "ProprietaryTotal": "float64",
    "ProprietaryBalance": "float64",

    "BrokerageSales": "float64",
    "BrokeragePurchases": "float64",
    "BrokerageTotal": "float64",
    "BrokerageBalance": "float64",

    "IndividualsSales": "float64",
    "IndividualsPurchases": "float64",
    "IndividualsTotal": "float64",
    "IndividualsBalance": "float64",

    "ForeignersSales": "float64",
    "ForeignersPurchases": "float64",
    "ForeignersTotal": "float64",
    "ForeignersBalance": "float64",

    "SecuritiesCosSales": "float64",
    "SecuritiesCosPurchases": "float64",
    "SecuritiesCosTotal": "float64",
    "SecuritiesCosBalance": "float64",

    "InvestmentTrustsSales": "float64",
    "InvestmentTrustsPurchases": "float64",
    "InvestmentTrustsTotal": "float64",
    "InvestmentTrustsBalance": "float64",

    "BusinessCosSales": "float64",
    "BusinessCosPurchases": "float64",
    "BusinessCosTotal": "float64",
    "BusinessCosBalance": "float64",

    "OtherInstitutionsSales": "float64",
    "OtherInstitutionsPurchases": "float64",
    "OtherInstitutionsTotal": "float64",
    "OtherInstitutionsBalance": "float64",

    "InsuranceCosSales": "float64",
    "InsuranceCosPurchases": "float64",
    "InsuranceCosTotal": "float64",
    "InsuranceCosBalance": "float64",

    "CityBKsRegionalBKsEtcSales": "float64",
    "CityBKsRegionalBKsEtcPurchase": "float64",
    "CityBKsRegionalBKsEtcTotal": "float64",
    "CityBKsRegionalBKsEtcBalance": "float64",

    "TrustBanksSales": "float64",
    "TrustBanksPurchases": "float64",
    "TrustBanksTotal": "float64",
    "TrustBanksBalance": "float64",

    "OtherFinancialInstitutionsSales": "float64",
    "OtherFinancialInstitutionsPurchases": "float64",
    "OtherFinancialInstitutionsTotal": "float64",
    "OtherFinancialInstitutionsBalance": "float64",
}

financials_dtypes: Mapping[Hashable, DtypeArg] = {
    "DisclosureNumber": "Int64",
    "DateCode": "string",
    "SecuritiesCode": "Int64",
    "TypeOfDocument": "string",
    "TypeOfCurrentPeriod": "string",

    "NetSales": "float64",
    "OperatingProfit": "float64",
    "OrdinaryProfit": "float64",
    "Profit": "float64",
    "EarningsPerShare": "float64",

    "TotalAssets": "float64",
    "Equity": "float64",
    "EquityToAssetRatio": "float64",
    "BookValuePerShare": "float64",

    # Forecasts
    "ForecastNetSales": "float64",
    "ForecastOperatingProfit": "float64",
    "ForecastOrdinaryProfit": "float64",
    "ForecastProfit": "float64",
    "ForecastEarningsPerShare": "float64",

    # Booleans (nullable!)
    "ApplyingOfSpecificAccountingOfTheQuarterlyFinancialStatements": "boolean",
    "MaterialChangesInSubsidiaries": "boolean",
    "ChangesBasedOnRevisionsOfAccountingStandard": "boolean",
    "ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard": "boolean",
    "ChangesInAccountingEstimates": "boolean",
    "RetrospectiveRestatement": "boolean",

    # Share counts
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": "Int64",
    "NumberOfTreasuryStockAtTheEndOfFiscalYear": "Int64",
    "AverageNumberOfShares": "Int64",
}

stock_list_dtypes: Mapping[Hashable, DtypeArg] = {
    "SecuritiesCode": "Int64",            # nullable integer
    "EffectiveDate": "string",            # YYYYMMDD, parse later if needed
    "Name": "string",
    "Section/Products": "string",
    "NewMarketSegment": "string",
    "33SectorCode": "Int64",
    "33SectorName": "string",
    "17SectorCode": "Int64",
    "17SectorName": "string",
    "NewIndexSeriesSizeCode": "Int64",
    "NewIndexSeriesSize": "string",
    "TradeDate": "string",                 # often numeric YYYYMMDD
    "Close": "Float64",
    "IssuedShares": "Float64",
    "MarketCapitalization": "Float64",
    "Universe0": "boolean"
}

NA_DASHES = [
    "-",        # ASCII hyphen
    "－",       # FULLWIDTH hyphen (U+FF0D) ← THIS IS YOUR CURRENT ERROR
    "–",        # en dash
    "—",        # em dash
]


# Read CSV
df_financials = pd.read_csv(
    csv_financials_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=financials_dtypes,
    parse_dates=["DateCode"],
    date_format={"DateCode": "%Y%m%d"},
)

df_options = pd.read_csv(
    csv_options_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=options_dtypes,
    parse_dates=["DateCode", "ContractMonth"],
    date_format={"DateCode": "%Y%m%d", "ContractMonth": "%Y%m"},
)

df_secondary_stock_prices = pd.read_csv(
    csv_secondary_stock_prices_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=secondary_stock_prices_dtypes,
    parse_dates=["Date"],
    date_format={"Date": "%Y-%m-%d"},
)

df_stock_prices = pd.read_csv(
    csv_stock_prices_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=stock_prices_dtypes,
    parse_dates=["Date"],
    date_format={"Date": "%Y-%m-%d"},
)

df_trades = pd.read_csv(
    csv_trades_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=trades_dtypes,
    parse_dates=["Date", "StartDate", "EndDate"],
    date_format={"Date": "%Y-%m-%d", "StartDate": "%Y-%m-%d", "EndDate": "%Y-%m-%d"},
)

df_stock_list = pd.read_csv(
    csv_stock_list_path,
    na_values=NA_DASHES,
    keep_default_na=True,
    dtype=stock_list_dtypes,
    parse_dates=["EffectiveDate", "TradeDate"],
    date_format={"EffectiveDate": "%Y%m%d", "TradeDate": "%Y%m%d"},
)

# Connect to SQLite
conn = sqlite3.connect(db_path)

# Write to SQLite table
df_financials.to_sql(
    name="financials",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_options.to_sql(
    name="options",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_secondary_stock_prices.to_sql(
    name="secondary_stock_prices",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_stock_prices.to_sql(
    name="stock_prices",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_trades.to_sql(
    name="trades",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

df_stock_list.to_sql(
    name="stock_list",
    con=conn,
    if_exists="replace",   # "replace", "append", or "fail"
    index=False
)

# Close connection
conn.close()
print("FINISHED")
