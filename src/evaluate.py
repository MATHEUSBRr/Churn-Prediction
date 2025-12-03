import os, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_OUT = os.path.join(BASE, "models", "rf_churn.joblib")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_artifact():
    data = joblib.load(MODEL_OUT)
    return data["model"], data["X_test"], data["y_test"]

def plot_roc(model, X_test, y_test, out_path):
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0,1],[0,1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion(model, X_test, y_test, out_path):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_feature_importance(model, X_test, out_path):
    importances = model.feature_importances_
    names = X_test.columns
    idx = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(8,6))
    plt.barh(names[idx][::-1], importances[idx][::-1])
    plt.xlabel("Importance")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

if __name__ == '__main__':
    model, X_test, y_test = load_artifact()
    plot_roc(model, X_test, y_test, os.path.join(FIG_DIR, "roc.png"))
    plot_confusion(model, X_test, y_test, os.path.join(FIG_DIR, "confusion.png"))
    plot_feature_importance(model, X_test, os.path.join(FIG_DIR, "feat_imp.png"))
    # print classification report
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Figures saved to", FIG_DIR)
