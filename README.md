# AutoDC — Data-Centric Quality Assessment Platform

AutoDC is a local-first Streamlit application inspired by the supplied AutoDC project report and HI-DBSF paper. It provides a working tabular data pipeline with optional image quality processing.

## Features

- Upload CSV, JSON, JSONL, Excel
- Normalize column names
- Dataset profile, missing values, duplicate rows, constant columns, outlier candidates
- Transparent prototype composite quality score (0–100)
- Optional KS two-sample drift checks against a reference dataset
- Hybrid Isolation Forest + DBSCAN anomaly scores
- Review-first cleaning: exact duplicates, empty rows/columns, median/mode imputation, optional z-score filtering, min-max scaling, undersampling, PCA
- Download processed CSV, JSON report, audit log, PDF report
- Optional ZIP image pipeline: blur proxy, review/skip blurry files, resize accepted images to 224×224
- Local processing; uploads are handled in the active app session

## Requirements

Python 3.10 or later recommended.

## Run locally

### Windows (PowerShell)

```powershell
cd AutoDC_Project
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
cd AutoDC_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the application in a browser, usually at `http://localhost:8501`.

## Workflow

1. Open **Upload & Profile** and upload a supported tabular dataset.
2. Use **Quality Assessment** to inspect missingness, duplicates, numeric outliers and optional drift.
3. Use **HI-DBSF Anomalies** to calculate scores. Anomalies are flagged for review and are not deleted automatically.
4. Use **Cleaning Workspace** to choose transformations and preview the processed data.
5. Use **Export & Reports** to download the processed CSV, quality JSON, audit JSON and PDF.
6. Optionally use **Image Quality** to upload a ZIP of images and create a 224×224 accepted-image ZIP.

## How the HI-DBSF-inspired module works

- Numeric feature columns are median-imputed and standardized.
- Isolation Forest uses 100 trees and a maximum subsample of 256.
- DBSCAN `min_samples = max(4, 2 × number_of_numeric_features)`.
- Epsilon is estimated from a k-nearest-neighbor distance quantile as a practical elbow proxy.
- The Isolation Forest anomaly scores are min-max normalized from `-score_samples`.
- For fewer than 1,000 records, fusion weight alpha is fixed at 0.5, following the paper's caution that adaptive fusion can be unstable on small datasets. For larger datasets, an adaptive weight is computed from DBSCAN noise and high Isolation Forest score fractions.
- Threshold is mean hybrid score + 1.5 standard deviations.

**Implementation note:** This is a runnable, HI-DBSF-inspired prototype, not a guarantee of exact numerical reproduction of the paper's path-length score or all parameters. The paper's formula uses a path-length score and k-NN elbow epsilon. The current implementation uses scikit-learn score_samples normalized to 0–1 and an automatic k-distance quantile proxy. Record these implementation choices accurately in academic reports.

## Composite quality score

The app reports six dimensions, each scored 0–100 and averaged equally:
- Completeness
- Consistency (duplicates and whitespace issues)
- Outlier quality
- Class balance (when a target is selected)
- Drift (neutral 100 until a baseline is provided)
- Image quality (neutral 100 in the tabular report; image processing has a separate report)

This is a transparent engineering score for the prototype, not a clinically validated or universally standardized metric. Domain-specific range checks need domain rules; AutoDC does not assume all numerical columns should lie between 1 and 10.

## Important ML and data safety notes

- The original upload is not overwritten.
- Anomaly scores do not prove that a row is erroneous. Review before deletion.
- Do not run undersampling, imputation, scaling, PCA or anomaly removal on the test set when conducting model evaluation. Fit learned preprocessing only on training folds.
- For the Breast Cancer Wisconsin example, flagged malignant-labelled records should be retained unless independent evidence proves a data error.
- This tool is for data preparation and research, not for medical diagnosis.

## Project structure

```text
AutoDC_Project/
├── app.py
├── requirements.txt
├── README.md
├── autodc/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── assessment.py
│   ├── hidbsf.py
│   ├── cleaning.py
│   ├── reporting.py
│   └── images.py
└── tests/
    ├── test_ingestion.py
    ├── test_assessment.py
    └── test_cleaning.py
```

## Research alignment

The supplied project report describes four main stages: ingestion and unification, quality assessment, cleaning and reduction, and output with monitoring. This MVP implements the local tabular workflow, optional reference-based KS drift checks, report/audit export, and an image ZIP quality module. A continuous filesystem watcher, multi-source directory orchestration, and persistent database/user authentication are not included in this first MVP.
