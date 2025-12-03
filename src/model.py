import os, joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed_churn.csv")
MODEL_OUT = os.path.join(BASE, "models", "rf_churn.joblib")
METRICS_OUT = os.path.join(BASE, "reports", "metrics.txt")

def load_processed(path=PROCESSED):
    return pd.read_csv(path)

def prepare_Xy(df):
    X = df.drop(columns=["customer_id","churn"])
    y = df["churn"]
    return X, y

def train_and_save():
    df = load_processed()
    X, y = prepare_Xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    param_grid = {"n_estimators":[50,100], "max_depth":[5,10,None]}
    rf = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring="f1")
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    joblib.dump({"model":best, "X_test":X_test, "y_test":y_test}, MODEL_OUT)
    preds = best.predict(X_test)
    proba = best.predict_proba(X_test)[:,1]
    report = classification_report(y_test, preds)
    roc = roc_auc_score(y_test, proba)
    with open(METRICS_OUT, "w") as f:
        f.write("Best params: " + str(gs.best_params_) + "\n\n")
        f.write("Classification report:\n" + report + "\n")
        f.write("ROC-AUC: " + str(roc) + "\n")
    print("Model saved to", MODEL_OUT)
    print("Metrics saved to", METRICS_OUT)

if __name__ == '__main__':
    train_and_save()
