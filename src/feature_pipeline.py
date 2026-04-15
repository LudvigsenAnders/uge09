
# src/feature_pipeline.py
from typing import List, Iterable, Tuple
import pandas as pd


class FeaturePipeline:
    """
    Feature pipeline with explicit, config-driven schema.

    - feature_cols_ is defined at init time
    - safe for training, backtest, and live
    """

    def __init__(
        self,
        group_col: str,
        base_cols: List[str],
        lags: Iterable[int],
    ):
        self.group_col = group_col
        self.base_cols = list(base_cols)
        self.lags = list(lags)
        self.feature_cols_: List[str] = (
            self.base_cols + [f"{col}_lag{lag}" for col in self.base_cols for lag in self.lags]
        )

    # -------------------------------------------------
    # Constructors
    # -------------------------------------------------
    @classmethod
    def from_config(cls, strategy_config: dict) -> "FeaturePipeline":
        """
        Build FeaturePipeline directly from strategy.yaml config.
        """
        return cls(
            group_col=strategy_config["group_col"],
            base_cols=strategy_config["features"]["base_cols"],
            lags=strategy_config["features"]["lags"],
        )

    # -------------------------------------------------
    # Training-only: lag creation
    # -------------------------------------------------
    def add_grouped_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TRAINING / FEATURE-BUILD ONLY.
        Creates lagged features on a panel.
        """
        for col in self.base_cols:
            for lag in self.lags:
                df[f"{col}_lag{lag}"] = (
                    df.groupby(self.group_col)[col].shift(lag)
                )
        return df

    def finalize_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        TRAINING / FEATURE-BUILD ONLY.

        Drops rows with incomplete lag history.
        """
        df = df.dropna(subset=self.feature_cols_).reset_index(drop=True)
        return df, self.feature_cols_

    # -------------------------------------------------
    # Safe for live: normalization
    # -------------------------------------------------
    def cross_sectional_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cross-sectional z-score normalization by Date.

        SAFE for live trading.
        """
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Missing required feature columns: {missing}"
            )

        df[self.feature_cols_] = (
            df.groupby("Date")[self.feature_cols_]
              .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
        )
        return df
