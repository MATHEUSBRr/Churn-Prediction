import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_IN = os.path.join(BASE, "data", "churn_data.csv")
PROCESSED = os.path.join(BASE, "data", "processed_churn.csv")

def load_data(path=DATA_IN):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    df["total_charges"] = df["total_charges"].clip(lower=0)
    return df

def feature_engineering(df):
    df = df.copy()
    df["avg_charge_per_month"] = df["total_charges"] / df["tenure"].replace(0,1)
    df["inactive_flag"] = (df["num_logins_last_30d"] == 0).astype(int)
    df = pd.get_dummies(df, columns=["contract_type","payment_method","gender"], drop_first=True)
    return df

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)
    df.to_csv(PROCESSED, index=False)
    print("Processed data saved to", PROCESSED)
