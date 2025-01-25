import pandas as pd
import numpy as np

def load_csv(file_path):
    return pd.read_csv(file_path)

def to_float_safe(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return np.nan

def map_categorical(data, mapping):
    return data.replace(mapping)

def preprocess(df, numeric_cols, cat_cols, target_col):
    for col in numeric_cols:
        df[col] = df[col].apply(to_float_safe)
    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes
    return df.dropna(), df[target_col].values