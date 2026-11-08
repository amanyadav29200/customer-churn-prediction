"""Small, transparent model explanation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def positive_probability(model, row: pd.DataFrame) -> float:
    """Return the probability for the positive (churn = 1) class."""
    classes = model.named_steps["classifier"].classes_
    index = int(np.flatnonzero(classes == 1)[0])
    return float(model.predict_proba(row)[:, index][0])


def local_reference_effects(bundle: dict, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Estimate one-feature-at-a-time effects versus typical training values.

    This is a local sensitivity display, not a causal explanation. Each feature
    is replaced by its training median (numeric) or mode (categorical), with all
    other submitted values held fixed.
    """
    model = bundle["model"]
    reference = bundle["reference_values"]
    baseline = positive_probability(model, row)
    effects = []
    for column in bundle["feature_columns"]:
        if column not in reference:
            continue
        changed = row.copy()
        changed.loc[:, column] = reference[column]
        alternate = positive_probability(model, changed)
        effects.append(
            {
                "feature": column,
                "submitted_value": row.iloc[0][column],
                "reference_value": reference[column],
                "risk_change_vs_reference": baseline - alternate,
            }
        )
    frame = pd.DataFrame(effects)
    if not frame.empty:
        frame["absolute_effect"] = frame["risk_change_vs_reference"].abs()
        frame = frame.sort_values("absolute_effect", ascending=False).drop(columns="absolute_effect")
    return baseline, frame
