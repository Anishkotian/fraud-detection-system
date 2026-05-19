"""
Script to generate a synthetic fraud transaction dataset for testing.
Run this once to create transactions.csv in the data/ folder.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 10000
FRAUD_RATE = 0.05  # 5% fraud — reflects real-world class imbalance

merchant_categories = ["grocery", "electronics", "restaurant", "travel", "clothing", "pharmacy", "entertainment", "fuel"]
locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego"]
device_types = ["mobile", "desktop", "tablet", "POS_terminal"]
transaction_types = ["online", "in_store", "ATM", "contactless"]

n_fraud = int(N * FRAUD_RATE)
n_legit = N - n_fraud

def generate_legit():
    return {
        "transaction_amount": round(np.random.lognormal(mean=3.5, sigma=1.0), 2),
        "transaction_time": random.randint(6, 23),  # legit: mostly daytime
        "merchant_category": random.choice(merchant_categories),
        "location": random.choice(locations),
        "device_type": random.choice(device_types),
        "transaction_type": random.choice(transaction_types),
        "account_age_days": random.randint(30, 3650),
        "num_transactions_today": random.randint(1, 5),
        "distance_from_home_km": round(random.uniform(0, 50), 2),
        "is_foreign_transaction": random.choices([0, 1], weights=[0.95, 0.05])[0],
        "is_fraud": 0
    }

def generate_fraud():
    return {
        "transaction_amount": round(np.random.lognormal(mean=5.5, sigma=1.2), 2),  # larger amounts
        "transaction_time": random.randint(0, 5),  # fraud: often late night
        "merchant_category": random.choice(["electronics", "travel", "entertainment"]),
        "location": random.choice(locations),
        "device_type": random.choice(["mobile", "desktop"]),
        "transaction_type": random.choice(["online", "ATM"]),
        "account_age_days": random.randint(1, 90),  # newer accounts more risky
        "num_transactions_today": random.randint(5, 20),  # many transactions in a day
        "distance_from_home_km": round(random.uniform(100, 5000), 2),  # far from home
        "is_foreign_transaction": random.choices([0, 1], weights=[0.4, 0.6])[0],
        "is_fraud": 1
    }

records = [generate_legit() for _ in range(n_legit)] + [generate_fraud() for _ in range(n_fraud)]
random.shuffle(records)

df = pd.DataFrame(records)
df.to_csv("transactions.csv", index=False)
print(f"Dataset created: {len(df)} rows, {df['is_fraud'].sum()} fraudulent ({df['is_fraud'].mean()*100:.1f}%)")
