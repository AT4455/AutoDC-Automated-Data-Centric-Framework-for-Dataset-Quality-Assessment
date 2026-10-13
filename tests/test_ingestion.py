from io import BytesIO
import pandas as pd
from autodc.ingestion import load_table, standardize_columns

def test_standardize_columns():
    df = pd.DataFrame({"Clump Thickness":[1], "Cell Size":[2], "Cell Size!":[3]})
    out = standardize_columns(df)
    assert list(out.columns) == ["clump_thickness", "cell_size", "cell_size_2"]

def test_load_csv():
    payload = BytesIO(b"A,B\n1,2\n3,?\n")
    df = load_table(payload, "demo.csv")
    assert df.shape == (2,2)
    assert df.iloc[1,1] != df.iloc[1,1]  # NaN
