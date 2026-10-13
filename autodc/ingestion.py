import json
import re
import pandas as pd

def standardize_columns(df):
    out = df.copy()
    names = []
    for col in out.columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower()).strip("_")
        if not name:
            name = "column"
        names.append(name)
    # ensure unique names
    seen = {}
    unique = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    out.columns = unique
    return out

def load_table(file_obj, filename=None):
    name = (filename or getattr(file_obj, "name", "")).lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file_obj, na_values=["?", "NA", "N/A", "null", "NULL", ""])
    elif name.endswith(".jsonl"):
        df = pd.read_json(file_obj, lines=True)
    elif name.endswith(".json"):
        try:
            df = pd.read_json(file_obj)
        except ValueError:
            file_obj.seek(0)
            df = pd.DataFrame(json.load(file_obj))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError("Supported tabular formats: CSV, JSON, JSONL, XLSX and XLS.")
    if df.empty:
        raise ValueError("The uploaded dataset has no rows.")
    return standardize_columns(df)
