import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def _estimate_eps(x, min_samples):
    if len(x) <= min_samples + 1:
        return 0.5
    nn = NearestNeighbors(n_neighbors=min(min_samples, len(x)-1))
    nn.fit(x)
    distances, _ = nn.kneighbors(x)
    kth = np.sort(distances[:, -1])
    # A robust elbow proxy: 90th percentile of k-distance. User can review scores.
    return float(max(0.15, np.quantile(kth, 0.90)))

def detect_hybrid_anomalies(df, target_col=None):
    features = df.drop(columns=[target_col], errors="ignore").select_dtypes(include=np.number)
    if features.shape[1] == 0:
        raise ValueError("No numeric feature columns are available for anomaly detection.")
    if len(features) < 5:
        raise ValueError("At least five records are required for hybrid anomaly detection.")
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(features)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    n, d = x.shape
    min_samples = max(4, 2*d)
    eps = _estimate_eps(x, min_samples)
    iso = IsolationForest(n_estimators=100, max_samples=min(256,n),
                          contamination="auto", random_state=42)
    iso.fit(x)
    # Convert sklearn's decision function to a stable 0-1 anomaly score.
    raw = -iso.score_samples(x)
    lo, hi = float(raw.min()), float(raw.max())
    if hi > lo:
        if_score = (raw-lo)/(hi-lo)
    else:
        if_score = np.zeros_like(raw)
    db_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x)
    db_noise = (db_labels == -1).astype(float)
    r_db = float(db_noise.mean())
    r_if = float((if_score > 0.5).mean())
    if n < 1000:
        alpha = 0.5
        note = "Fixed fusion weight α=0.5 used because the paper notes adaptive fusion can be unstable for datasets smaller than 1,000 records."
    else:
        ratio = r_db / max(r_if, 1e-8)
        alpha = float(np.clip(1.0-ratio, 0.1, 0.9))
        note = "Adaptive fusion weight calculated from DBSCAN noise fraction and Isolation Forest high-score fraction."
    fusion = alpha*if_score + (1-alpha)*db_noise
    threshold = float(np.mean(fusion) + 1.5*np.std(fusion))
    flagged = fusion > threshold
    out = df.copy()
    out["isolation_forest_score"] = if_score
    out["dbscan_noise"] = db_noise.astype(bool)
    out["hybrid_score"] = fusion
    out["flagged_for_review"] = flagged
    meta = {"alpha": alpha, "eps": eps, "min_samples": min_samples,
            "threshold": threshold, "dbscan_noise_count": int(db_noise.sum()),
            "flagged_count": int(flagged.sum()), "note": note,
            "method_note": "Isolation Forest scores are min-max normalized sklearn anomaly scores; this is a practical approximation and not the paper's exact path-length formula implementation."}
    return out, meta
