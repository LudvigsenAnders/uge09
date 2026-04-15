
# src/model.py
import pandas as pd


class AlphaModel:
    def __init__(self, model, name: str | None = None):
        self.model = model
        self.name = name or model.__class__.__name__

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def score(self, X: pd.DataFrame) -> pd.Series:
        scores = self.model.predict_proba(X)[:, 1]
        return pd.Series(scores, index=X.index)
