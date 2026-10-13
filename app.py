from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px

from autodc.ingestion import load_table, standardize_columns
from autodc.assessment import assess_dataset, compare_drift
from autodc.cleaning import clean_dataset
from autodc.hidbsf import detect_hybrid_anomalies
from autodc.reporting import make_pdf_report, json_safe

APP_TITLE = "AutoDC | Data-Centric Quality Studio"
st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#12263a,#235789); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0; font-size: 2.15rem;}
.hero p {color: #dbeafe; margin: .45rem 0 0 0;}
.small-muted {color: #64748b; font-size: .88rem;}
div[data-testid="stMetric"] {border: 1px solid #e2e8f0; padding: 12px; border-radius: 12px;}
</style>
<div class="hero"><h1>AutoDC</h1><p>Automated Data-Centric Framework for Dataset Quality Assessment</p></div>
""", unsafe_allow_html=True)

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "report" not in st.session_state:
    st.session_state.report = None
if "anomaly_df" not in st.session_state:
    st.session_state.anomaly_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "audit" not in st.session_state:
    st.session_state.audit = []

with st.sidebar:
    st.title("Workspace")
    page = st.radio("Navigate", ["1 · Upload & Profile", "2 · Quality Assessment",
                                 "3 · HI-DBSF Anomalies", "4 · Cleaning Workspace",
                                 "5 · Export & Reports", "6 · Image Quality (optional)"])
    st.divider()
    st.caption("Local processing • No cloud upload by AutoDC")
    st.caption("Prototype version 1.0")

if page == "1 · Upload & Profile":
    st.subheader("Stage 1 — Ingestion and Unification")
    st.write("Upload a CSV, JSON, JSON Lines, or Excel file. The original upload is kept unchanged in session memory.")
    upload = st.file_uploader("Choose a dataset", type=["csv", "json", "jsonl", "xlsx", "xls"])
    if upload is not None:
        try:
            df = load_table(upload, upload.name)
            st.session_state.raw_df = df.copy()
            st.session_state.source_name = upload.name
            st.session_state.report = None
            st.session_state.anomaly_df = None
            st.session_state.cleaned_df = None
            st.session_state.audit = [{
                "action": "INGEST_DATASET", "details": f"Loaded {upload.name}",
                "rows_before": int(len(df)), "rows_after": int(len(df))
            }]
            st.success(f"Loaded {upload.name}")
        except Exception as e:
            st.error(f"Could not read this file: {e}")

    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        c4.metric("Exact duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.markdown("#### Data preview")
        st.dataframe(df.head(100), use_container_width=True)
        with st.expander("Column types and memory overview"):
            info = pd.DataFrame({"column": df.columns.astype(str),
                                 "dtype": [str(x) for x in df.dtypes],
                                 "non_null": [int(x) for x in df.notna().sum()],
                                 "missing": [int(x) for x in df.isna().sum()],
                                 "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]})
            st.dataframe(info, use_container_width=True, hide_index=True)
        if st.button("Standardize column names", type="primary"):
            before = list(df.columns)
            st.session_state.raw_df = standardize_columns(df)
            st.session_state.audit.append({"action":"STANDARDIZE_COLUMNS",
                                           "details": f"Normalized column names from {before}",
                                           "rows_before":len(df),"rows_after":len(df)})
            st.success("Column names standardized. Review the preview before continuing.")
    else:
        st.info("Upload a dataset to start the AutoDC pipeline.")

elif page == "2 · Quality Assessment":
    st.subheader("Stage 2 — Quality Assessment")
    df = st.session_state.raw_df
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        target = st.selectbox("Target / label column (optional)", ["(None)"] + list(df.columns))
        target_col = None if target == "(None)" else target
        baseline_file = st.file_uploader("Optional reference/baseline dataset for drift checks", type=["csv","json","jsonl","xlsx","xls"], key="baseline")
        baseline_df = None
        if baseline_file:
            try:
                baseline_df = load_table(baseline_file, baseline_file.name)
                st.caption(f"Reference dataset loaded: {len(baseline_df)} rows")
            except Exception as e:
                st.error(f"Could not read baseline: {e}")
        report = assess_dataset(df, target_col=target_col)
        if baseline_df is not None:
            report["drift"] = compare_drift(baseline_df, df)
        st.session_state.report = report
        st.markdown("#### Quality overview")
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Composite score", f"{report['quality_score']:.1f}/100")
        q2.metric("Missing cells", f"{report['missing_cells']:,}")
        q3.metric("Duplicate rows", f"{report['duplicate_rows']:,}")
        q4.metric("Outlier candidates", f"{report['outlier_rows']:,}")
        st.progress(max(0.0,min(1.0,report["quality_score"]/100)))
        st.caption("Composite score is an implementation-defined diagnostic score; see README for component weights and limitations.")
        scores = pd.DataFrame({"Dimension": list(report["dimension_scores"].keys()),
                               "Score": list(report["dimension_scores"].values())})
        st.plotly_chart(px.bar(scores, x="Dimension", y="Score", range_y=[0,100],
                               title="Quality dimension scores (higher is better)"),
                        use_container_width=True)
        left,right = st.columns(2)
        with left:
            st.markdown("#### Missing values by column")
            st.dataframe(pd.DataFrame(report["missing_by_column"]), use_container_width=True, hide_index=True)
        with right:
            st.markdown("#### Dataset structure")
            st.dataframe(pd.DataFrame(report["column_profile"]), use_container_width=True, hide_index=True)
        if target_col:
            st.markdown("#### Target class distribution")
            dist = df[target_col].astype("string").fillna("<missing>").value_counts().rename_axis("class").reset_index(name="count")
            st.plotly_chart(px.bar(dist, x="class", y="count", title="Target class counts"), use_container_width=True)
        if report.get("drift"):
            st.markdown("#### Numeric drift checks (KS test)")
            st.dataframe(pd.DataFrame(report["drift"]), use_container_width=True, hide_index=True)
        st.download_button("Download quality report (JSON)",
                           json.dumps(json_safe(report), indent=2),
                           file_name="autodc_quality_report.json", mime="application/json")

elif page == "3 · HI-DBSF Anomalies":
    st.subheader("Stage 3A — HI-DBSF anomaly review")
    df = st.session_state.raw_df
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        target = st.selectbox("Target / label column to exclude from anomaly features", ["(None)"] + list(df.columns), key="anom_target")
        target_col = None if target == "(None)" else target
        st.warning("Anomaly scores identify unusual records, not confirmed errors. AutoDC does not delete anomalies automatically.")
        if st.button("Run hybrid anomaly detector", type="primary"):
            with st.spinner("Fitting Isolation Forest and DBSCAN..."):
                try:
                    result, meta = detect_hybrid_anomalies(df, target_col=target_col)
                    st.session_state.anomaly_df = result
                    st.session_state.anomaly_meta = meta
                    st.session_state.audit.append({"action":"HI_DBSF_ANALYSIS",
                        "details":f"Flagged {int(result['flagged_for_review'].sum())} records; records retained",
                        "rows_before":len(df),"rows_after":len(df)})
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")
        if st.session_state.anomaly_df is not None:
            result = st.session_state.anomaly_df
            meta = st.session_state.get("anomaly_meta", {})
            a,b,c,d = st.columns(4)
            a.metric("Rows evaluated", f"{len(result):,}")
            b.metric("Flagged for review", f"{int(result['flagged_for_review'].sum()):,}")
            c.metric("DBSCAN noise", f"{int(result['dbscan_noise'].sum()):,}")
            d.metric("Fusion weight α", f"{meta.get('alpha',0):.3f}")
            st.caption(meta.get("note",""))
            st.dataframe(result[result["flagged_for_review"]].head(500), use_container_width=True)
            st.markdown("#### All records with scores")
            st.dataframe(result, use_container_width=True)
            st.download_button("Download anomaly scores CSV", result.to_csv(index=False),
                               file_name="autodc_anomaly_scores.csv", mime="text/csv")

elif page == "4 · Cleaning Workspace":
    st.subheader("Stage 3B — Cleaning and reduction")
    df = st.session_state.raw_df
    if df is None:
        st.warning("Upload a dataset first.")
    else:
        st.write("Choose operations deliberately. The original dataset remains unchanged; cleaning creates a new processed copy.")
        target = st.selectbox("Target / label column (optional)", ["(None)"] + list(df.columns), key="clean_target")
        target_col = None if target == "(None)" else target
        c1,c2 = st.columns(2)
        with c1:
            drop_duplicates = st.checkbox("Remove exact duplicate rows", value=True)
            impute_missing = st.checkbox("Impute missing values (numeric median / categorical mode)", value=True)
            remove_junk = st.checkbox("Remove all-empty rows and columns", value=True)
            normalize = st.checkbox("Min-max normalize numeric feature columns", value=False)
        with c2:
            remove_extreme = st.checkbox("Remove extreme numeric rows (|z| > 5)", value=False)
            balance = st.checkbox("Balance target by undersampling majority class", value=False)
            do_pca = st.checkbox("Apply PCA retaining 95% variance", value=False)
            st.caption("Normalization, undersampling and PCA can change model semantics. Use only when appropriate for your task.")
        if balance and target_col is None:
            st.warning("Select a target column before enabling class balancing.")
        if do_pca and target_col is not None:
            st.warning("PCA replaces feature columns with components; target label is preserved separately.")
        if st.button("Preview and run cleaning", type="primary", disabled=(balance and target_col is None)):
            try:
                cleaned, audit, details = clean_dataset(
                    df, target_col=target_col, drop_duplicates=drop_duplicates,
                    impute_missing=impute_missing, remove_junk=remove_junk,
                    normalize=normalize, remove_extreme=remove_extreme,
                    balance=balance, do_pca=do_pca
                )
                st.session_state.cleaned_df = cleaned
                st.session_state.clean_details = details
                st.session_state.audit.extend(audit)
                st.success("Cleaning completed. Review the before/after counts before exporting.")
            except Exception as e:
                st.error(f"Cleaning failed: {e}")
        if st.session_state.cleaned_df is not None:
            cleaned = st.session_state.cleaned_df
            details = st.session_state.get("clean_details", {})
            a,b,c,d = st.columns(4)
            a.metric("Original rows", f"{len(df):,}")
            b.metric("Processed rows", f"{len(cleaned):,}", delta=f"{len(cleaned)-len(df):,}")
            c.metric("Remaining missing cells", f"{int(cleaned.isna().sum().sum()):,}")
            d.metric("Remaining exact duplicates", f"{int(cleaned.duplicated().sum()):,}")
            st.markdown("#### Processed preview")
            st.dataframe(cleaned.head(100), use_container_width=True)
            st.markdown("#### Cleaning actions")
            st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True)
            st.download_button("Download processed CSV", cleaned.to_csv(index=False),
                               file_name="autodc_processed.csv", mime="text/csv")
            st.download_button("Download processing log JSON",
                               json.dumps(json_safe(st.session_state.audit), indent=2),
                               file_name="autodc_audit_log.json", mime="application/json")

elif page == "5 · Export & Reports":
    st.subheader("Stage 4 — Output, reports and audit trail")
    if st.session_state.raw_df is None:
        st.warning("Upload a dataset first.")
    else:
        raw = st.session_state.raw_df
        processed = st.session_state.cleaned_df
        if processed is None:
            st.info("No cleaned dataset exists yet. You can still export the original data quality report.")
            processed = raw
        report = st.session_state.report or assess_dataset(raw)
        report["source_file"] = st.session_state.source_name
        report["original_rows"] = int(len(raw))
        report["processed_rows"] = int(len(processed))
        report["audit_log"] = st.session_state.audit
        st.write("Downloads are generated locally in your browser session.")
        st.download_button("Download processed dataset CSV", processed.to_csv(index=False),
                           file_name="autodc_processed.csv", mime="text/csv")
        st.download_button("Download report JSON", json.dumps(json_safe(report), indent=2),
                           file_name="autodc_report.json", mime="application/json")
        st.download_button("Download audit log JSON", json.dumps(json_safe(st.session_state.audit), indent=2),
                           file_name="autodc_audit_log.json", mime="application/json")
        try:
            pdf = make_pdf_report(report, st.session_state.audit)
            st.download_button("Download PDF quality report", pdf,
                               file_name="autodc_quality_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"PDF report could not be generated: {e}")
        st.markdown("#### Current workflow summary")
        st.write({
            "source_file": st.session_state.source_name,
            "original_rows": int(len(raw)),
            "processed_rows": int(len(processed)),
            "missing_cells_before": int(raw.isna().sum().sum()),
            "missing_cells_after": int(processed.isna().sum().sum()),
            "audit_actions": len(st.session_state.audit)
        })
        st.caption("All operations are logged for traceability. Keep the original file to reproduce or audit the transformations.")

else:
    st.subheader("Optional image quality pipeline")
    st.write("This module scans images in a ZIP archive using Laplacian variance, marks blurry files for review, and resizes accepted images to 224×224. The archive is processed locally.")
    zip_file = st.file_uploader("Upload a ZIP containing image files", type=["zip"], key="images_zip")
    threshold = st.slider("Blurriness threshold (Laplacian variance)", min_value=0.0, max_value=500.0, value=100.0, step=5.0)
    if zip_file and st.button("Analyze and prepare image archive", type="primary"):
        from autodc.images import process_image_zip
        try:
            processed_zip, image_report = process_image_zip(zip_file, threshold=threshold)
            st.success(f"Processed {image_report['total_images']} images; flagged {image_report['flagged_blurry']} as blurry.")
            st.json(image_report)
            st.download_button("Download accepted images (224×224 ZIP)", processed_zip,
                               file_name="autodc_images_224.zip", mime="application/zip")
            st.download_button("Download image report JSON", json.dumps(json_safe(image_report), indent=2),
                               file_name="autodc_image_report.json", mime="application/json")
        except Exception as e:
            st.error(f"Image processing failed: {e}")

st.divider()
st.caption("AutoDC research prototype • Inspect cleaning decisions before using processed data for training.")
