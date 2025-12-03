import os
import pandas as pd
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.dirname(__file__))
PROCESSED = os.path.join(BASE, "data", "processed_churn.csv")
FIG_DIR = os.path.join(BASE, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def run_eda():
    df = pd.read_csv(PROCESSED)
    plt.figure()
    df['churn'].value_counts().plot(kind='bar')
    plt.title('Churn distribution (0=no,1=yes)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'churn_dist.png'))
    plt.close()
    plt.figure()
    plt.hist(df['age'], bins=20)
    plt.title('Age distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'age_hist.png'))
    plt.close()
    plt.figure(figsize=(6,4))
    plt.scatter(df['tenure'], df['monthly_charges'], alpha=0.3, s=10)
    plt.xlabel('Tenure (months)')
    plt.ylabel('Monthly charges')
    plt.title('Tenure vs Monthly Charges')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'tenure_vs_monthly.png'))
    plt.close()
    print('EDA figures saved to', FIG_DIR)

if __name__ == '__main__':
    run_eda()
