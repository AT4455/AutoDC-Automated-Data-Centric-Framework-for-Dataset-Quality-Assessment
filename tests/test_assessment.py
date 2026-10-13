import pandas as pd
from autodc.assessment import assess_dataset

def test_quality_report_detects_missing_duplicate():
    df = pd.DataFrame({"x":[1,2,2,None], "label":[0,1,1,0]})
    report = assess_dataset(df, target_col="label")
    assert report["rows"] == 4
    assert report["missing_cells"] == 1
    assert report["duplicate_rows"] == 1
    assert 0 <= report["quality_score"] <= 100
