"""Dataset loading and conservative cleaning for the IBM Telco sample."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COLUMN = "customerID"


def load_and_clean_data(path: str | Path = DEFAULT_DATA_PATH) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the real CSV and return features, binary target, and cleaned rows.

    Blank ``TotalCharges`` values are converted to missing values and then
    imputed inside the model pipeline. The customer identifier is kept in the
    returned cleaned frame for reporting but excluded from model features.
    Outliers are deliberately flagged during EDA rather than automatically
    removed: unusually high bills can be real customer behavior.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. See data/raw/README.md for the IBM source."
        )

    df = pd.read_csv(data_path)
    df.columns = [column.strip() for column in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {data_path.name}.")

    df = df.drop_duplicates().copy()
    if ID_COLUMN in df.columns:
        # In case a customer appears more than once, retain the first snapshot.
        df = df.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()

    if "TotalCharges" in df.columns:
        # IBM stores this field as text; 11 tenure-zero customers have blanks.
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce"
        )

    df[TARGET] = df[TARGET].astype("string").str.strip().str.lower().map({"no": 0, "yes": 1})
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype("int8")

    excluded = [TARGET]
    if ID_COLUMN in df.columns:
        excluded.append(ID_COLUMN)
    X = df.drop(columns=excluded).copy()
    y = df[TARGET].copy()
    return X, y, df


def feature_schema(X: pd.DataFrame) -> dict:
    """Return UI-friendly type and allowed-value information for raw features."""
    schema = {}
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_numeric_dtype(series):
            schema[column] = {
                "kind": "numeric",
                "minimum": float(series.min(skipna=True)),
                "maximum": float(series.max(skipna=True)),
                "median": float(series.median(skipna=True)),
                "integer": pd.api.types.is_integer_dtype(series),
            }
        else:
            values = sorted(series.dropna().astype(str).unique().tolist())
            schema[column] = {
                "kind": "categorical",
                "values": values,
                "default": str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else "",
            }
    return schema
