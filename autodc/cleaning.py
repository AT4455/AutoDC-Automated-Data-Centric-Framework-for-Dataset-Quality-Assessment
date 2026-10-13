import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def clean_dataset(df, target_col=None, drop_duplicates=True, impute_missing=True,
                  remove_junk=True, normalize=False, remove_extreme=False,
                  balance=False, do_pca=False):
    original = df.copy()
    out = df.copy()
    audit = []
    def log(action, details, before, after):
        audit.append({"action": action, "details": details,
                      "rows_before": int(before), "rows_after": int(after)})
    if remove_junk:
        before = len(out)
        empty_cols = [c for c in out.columns if out[c].isna().all()]
        if empty_cols:
            out = out.drop(columns=empty_cols)
        empty_rows = out.isna().all(axis=1)
        out = out.loc[~empty_rows].copy()
        log("REMOVE_ALL_EMPTY", f"Removed {len(empty_cols)} all-empty columns and {int(empty_rows.sum())} all-empty rows.", before, len(out))
        if target_col in empty_cols:
            target_col = None
    if drop_duplicates:
        before = len(out)
        count = int(out.duplicated().sum())
        out = out.drop_duplicates().copy()
        log("REMOVE_EXACT_DUPLICATES", f"Removed {count} exact duplicate rows.", before, len(out))
    if remove_extreme:
        numeric = out.select_dtypes(include=np.number).drop(columns=[target_col], errors="ignore")
        if not numeric.empty:
            z = (numeric-numeric.mean())/numeric.std(ddof=0).replace(0, np.nan)
            mask = z.abs().gt(5).any(axis=1).fillna(False)
            before = len(out)
            out = out.loc[~mask].copy()
            log("REMOVE_EXTREME_OUTLIERS", f"Removed {int(mask.sum())} rows with any numeric |z| > 5; review this choice for your domain.", before, len(out))
    if impute_missing:
        before_missing = int(out.isna().sum().sum())
        for col in out.columns:
            if not out[col].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                value = out[col].median()
                out[col] = out[col].fillna(value)
                strategy = f"median ({value})"
            else:
                mode = out[col].mode(dropna=True)
                value = mode.iloc[0] if len(mode) else "Unknown"
                out[col] = out[col].fillna(value)
                strategy = f"mode ({value})"
            audit.append({"action":"IMPUTE_MISSING", "details":f"{col}: {strategy}; missing cells before step={before_missing}",
                          "rows_before":len(out),"rows_after":len(out)})
    feature_cols = [c for c in out.columns if c != target_col]
    numeric_cols = out[feature_cols].select_dtypes(include=np.number).columns.tolist()
    if normalize and numeric_cols:
        scaler = MinMaxScaler()
        out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
        log("MIN_MAX_NORMALIZE", f"Normalized {len(numeric_cols)} numeric feature columns to [0,1].", len(out), len(out))
    if balance:
        if not target_col or target_col not in out.columns:
            raise ValueError("Choose a valid target column before balancing.")
        counts = out[target_col].value_counts(dropna=True)
        if len(counts) > 1:
            n_min = int(counts.min())
            pieces = [g.sample(n=n_min, random_state=42) for _,g in out.groupby(target_col, dropna=False)]
            before = len(out)
            out = pd.concat(pieces).sample(frac=1, random_state=42)
            log("UNDERSAMPLE_MAJORITY", f"Undersampled each class to {n_min} rows.", before, len(out))
    if do_pca:
        if not numeric_cols:
            raise ValueError("PCA requires numeric feature columns.")
        # PCA must not include target; impute any remaining values before fitting.
        x = out[numeric_cols].copy()
        x = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=numeric_cols, index=out.index)
        x = MinMaxScaler().fit_transform(x)
        pca = PCA(n_components=0.95, random_state=42)
        components = pca.fit_transform(x)
        comp_df = pd.DataFrame(components, index=out.index,
                               columns=[f"pc_{i+1}" for i in range(components.shape[1])])
        other_cols = [c for c in out.columns if c not in numeric_cols]
        out = pd.concat([comp_df, out[other_cols]], axis=1)
        log("PCA_REDUCTION", f"Retained {components.shape[1]} components for >=95% variance.", len(out), len(out))
    if not audit:
        log("NO_CHANGES", "No selected cleaning operation changed the dataset.", len(original), len(out))
    return out.reset_index(drop=True), audit, {"original_rows":len(original),"processed_rows":len(out)}
