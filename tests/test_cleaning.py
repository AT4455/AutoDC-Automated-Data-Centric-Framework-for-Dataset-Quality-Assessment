import pandas as pd
from autodc.cleaning import clean_dataset

def test_clean_duplicates_and_impute():
    df = pd.DataFrame({"x":[1.0,2.0,2.0,None], "label":[0,1,1,0]})
    cleaned, audit, details = clean_dataset(df, target_col="label",
        drop_duplicates=True, impute_missing=True, remove_junk=True,
        normalize=False, remove_extreme=False, balance=False, do_pca=False)
    assert len(cleaned) == 3
    assert cleaned.isna().sum().sum() == 0
    assert len(audit) >= 1
