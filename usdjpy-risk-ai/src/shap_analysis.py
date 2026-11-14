"""TreeExplainer-based local explanation helpers for promoted model reports."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def shap_top_drivers(
    model: object,
    row: pd.DataFrame,
    feature_order: Sequence[str],
    mean_abs_shap: pd.Series,
) -> list[str]:
    """Return top three per-row absolute SHAP drivers with stable tie-breaking."""
    if len(row) != 1 or list(row[feature_order].columns) != list(feature_order):
        raise ValueError("SHAP_FEATURE_SCHEMA_MISMATCH")
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP_UNAVAILABLE") from exc
    values = shap.TreeExplainer(model).shap_values(row[list(feature_order)])
    if isinstance(values, list):
        values = values[-1]
    absolute = pd.Series(np.abs(np.asarray(values)[0]), index=list(feature_order))
    ranking = pd.DataFrame({"sample": absolute, "mean": mean_abs_shap.reindex(feature_order).fillna(0.0)})
    return ranking.sort_values(["sample", "mean"], ascending=False, kind="stable").head(3).index.tolist()
