"""Production promotion gate; no partial pass can modify production state."""
from __future__ import annotations

from collections.abc import Mapping

REQUIRED_GATES = ("future_leakage", "feature_integrity", "feature_schema", "manifest_integrity", "oci_upload", "probability_ordering", "walk_forward", "calibration_saved", "psi_baseline_saved")


def production_accepted(metrics: Mapping[str, float | bool | int]) -> tuple[bool, list[str]]:
    """Return acceptance and every failed gate without promoting anything."""
    failures = [name for name in REQUIRED_GATES if metrics.get(name) is not True]
    if float(metrics.get("auc_roc", 0)) < 0.60: failures.append("auc_roc")
    if float(metrics.get("pr_auc", 0)) <= float(metrics.get("pr_auc_baseline", 0)): failures.append("pr_auc")
    if float(metrics.get("brier_score", 1)) > 0.25: failures.append("brier_score")
    if float(metrics.get("ece", 1)) > 0.05: failures.append("ece")
    if int(metrics.get("positive_count_total", 0)) < 100: failures.append("positive_count_total")
    if int(metrics.get("positive_count_min_fold", 0)) < 20: failures.append("positive_count_min_fold")
    return not failures, failures
