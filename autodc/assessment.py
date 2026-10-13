import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def _score_from_rate(rate):
    return float(np.clip(100.0 * (1.0 - float(rate)), 0, 100))

def assess_dataset(df, target_col=None):
    n, p = df.shape
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(1, n * p)
    missing_rate = missing_cells / total_cells
    duplicate_rows = int(df.duplicated().sum())
    duplicate_rate = duplicate_rows / max(1, n)
    numeric = df.select_dtypes(include=np.number)
    outlier_mask = pd.Series(False, index=df.index)
    for col in numeric.columns:
        x = pd.to_numeric(df[col], errors="coerce")
        sd = x.std(skipna=True)
        if pd.notna(sd) and sd > 0:
            z = ((x - x.mean()) / sd).abs()
            outlier_mask |= z.gt(3).fillna(False)
    outlier_rate = float(outlier_mask.mean()) if n else 0.0
    consistency_issues = 0
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].dropna().astype(str)
            if len(s):
                consistency_issues += int((s.str.strip() != s).sum())
    consistency_rate = min(1.0, (duplicate_rate + consistency_issues/max(1,n*p)))
    imbalance_rate = 0.0
    class_distribution = {}
    if target_col and target_col in df.columns:
        counts = df[target_col].value_counts(dropna=True)
        class_distribution = {str(k): int(v) for k,v in counts.items()}
        if len(counts) > 1:
            imbalance_rate = float(counts.max()/counts.sum())
    # Dataset-level drift needs a reference dataset; neutral score until one is provided.
    drift_score = 100.0
    dimensions = {
        "completeness": _score_from_rate(missing_rate),
        "consistency": _score_from_rate(consistency_rate),
        "outlier_quality": _score_from_rate(outlier_rate),
        "class_balance": _score_from_rate(max(0.0, imbalance_rate - 0.5) * 2) if class_distribution else 100.0,
        "drift": drift_score,
        "image_quality": 100.0
    }
    quality = float(np.mean(list(dimensions.values())))
    profile = []
    for col in df.columns:
        s = df[col]
        profile.append({
            "column": str(col), "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean()*100), 3),
            "unique": int(s.nunique(dropna=True)), "constant": bool(s.nunique(dropna=True) <= 1)
        })
    missing_by = [{"column": str(c), "missing_count": int(df[c].isna().sum()),
                   "missing_pct": round(float(df[c].isna().mean()*100), 3)}
                  for c in df.columns]
    return {
        "rows": int(n), "columns": int(p), "missing_cells": missing_cells,
        "missing_rate_pct": round(missing_rate*100, 4), "duplicate_rows": duplicate_rows,
        "outlier_rows": int(outlier_mask.sum()), "invalid_range_values": 0,
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
        "quality_score": round(quality, 2), "dimension_scores": dimensions,
        "missing_by_column": missing_by, "column_profile": profile,
        "class_distribution": class_distribution,
        "notes": [
            "The 0-100 score is a transparent prototype score, not a validated clinical or universal data-quality measure.",
            "Range violations require domain-specific bounds; AutoDC does not assume every numeric column must be 1-10.",
            "Drift score is neutral until a reference dataset is supplied."
        ]
    }

def compare_drift(reference, current):
    rows = []
    common = [c for c in reference.columns if c in current.columns]
    for col in common:
        a = pd.to_numeric(reference[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(a) >= 2 and len(b) >= 2:
            stat, pvalue = ks_2samp(a, b)
            rows.append({"column": str(col), "ks_statistic": float(stat),
                         "p_value": float(pvalue), "drift_detected_p_lt_0_05": bool(pvalue < 0.05),
                         "reference_n": int(len(a)), "current_n": int(len(b))})
    return rows
