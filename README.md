# 🛡️ FraudGuard AI — Fraud Transaction Detection System

A production-ready, end-to-end Machine Learning system that detects fraudulent banking transactions in real-time.
Built with Python, Streamlit, and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What This Project Does

| Feature | Description |
|---|---|
| 📂 Data Upload | Upload any CSV transaction dataset |
| ⚙️ Preprocessing | Auto-handles missing values, encoding, scaling |
| ⚖️ SMOTE Balancing | Fixes class imbalance so models learn fraud patterns |
| 🤖 3 ML Models | Logistic Regression, Random Forest, Isolation Forest |
| 📊 Evaluation | Confusion matrix, ROC-AUC, Precision/Recall, F1 |
| 🔍 Real-Time Prediction | Submit individual transactions, get instant risk score |
| 📦 Batch Prediction | Upload CSV → get predictions + download results |
| 🎚️ Threshold Tuning | Adjust decision threshold to balance precision/recall |
| 📈 Analytics Dashboard | Fraud trends, distributions, feature importances |
| 📚 Interview Prep | Built-in Q&A guide explaining all ML concepts |

---

## 🗂️ Project Structure

```
fraud-detection-system/
│
├── app.py              # Streamlit frontend — all 6 pages
├── train.py            # Model training (LR, RF, Isolation Forest)
├── predict.py          # Single + batch prediction with risk tagging
├── preprocessing.py    # Data pipeline: clean → encode → scale → SMOTE
├── utils.py            # All Plotly chart functions
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container deployment
├── .gitignore
├── README.md
│
├── models/
│   ├── logistic_model.pkl
│   ├── random_forest.pkl
│   └── isolation_forest.pkl
│
└── data/
    ├── generate_data.py   # Synthetic dataset generator
    └── transactions.csv   # Generated dataset
```

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate sample dataset
```bash
cd data
python generate_data.py
cd ..
```

### 5. Run the app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

---

## 📊 Dataset Format

Your CSV should contain these columns (others will be auto-detected):

| Column | Type | Example |
|---|---|---|
| `transaction_amount` | float | 250.00 |
| `transaction_time` | int | 14 (hour 0-23) |
| `merchant_category` | str | "electronics" |
| `location` | str | "New York" |
| `device_type` | str | "mobile" |
| `transaction_type` | str | "online" |
| `account_age_days` | int | 365 |
| `num_transactions_today` | int | 3 |
| `distance_from_home_km` | float | 5.0 |
| `is_foreign_transaction` | int | 0 or 1 |
| `is_fraud` | int | **0 or 1 (target)** |

---

## 🧠 ML Pipeline

```
Raw CSV
  │
  ▼
Handle Missing Values (median/mode imputation)
  │
  ▼
Label Encode Categoricals (merchant_category, device_type, etc.)
  │
  ▼
Train/Test Split (80/20, stratified)
  │
  ▼
StandardScaler (fit on train only — no leakage)
  │
  ▼
SMOTE on Training Set (synthetic fraud samples)
  │
  ▼
Train 3 Models in Parallel:
  ├── Logistic Regression (baseline, interpretable)
  ├── Random Forest (best performer, feature importances)
  └── Isolation Forest (anomaly detection, no labels needed)
  │
  ▼
Evaluate on Unbalanced Test Set (real-world conditions)
  │
  ▼
Confusion Matrix + ROC-AUC + Precision/Recall + F1
  │
  ▼
Save models as .pkl → Serve via Streamlit
```

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fraudguard .

# Run container
docker run -p 8501:8501 fraudguard

# Open http://localhost:8501
```

---

## ☁️ Streamlit Cloud Deployment

1. Push to GitHub (ensure `requirements.txt` is present)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy**

> Note: Streamlit Cloud free tier has memory limits. Use the sample dataset (10K rows) rather than very large CSVs.

---

## 🚢 Render Deployment

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Build Command**: `pip install -r requirements.txt && python data/generate_data.py`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**

---

## 📤 GitHub Push Commands

```bash
git init
git add .
git commit -m "feat: initial commit — FraudGuard AI fraud detection system"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-system.git
git push -u origin main
```

---

## 🔬 Key ML Concepts

### Why SMOTE?
With 95% legit / 5% fraud, a model predicting "all legit" scores 95% accuracy but catches 0 frauds.
SMOTE creates synthetic fraud samples in feature space to give the model balanced training signal.

### Why Precision AND Recall?
- **False Positive** (predict fraud, actually legit): Customer card blocked → friction, calls to support
- **False Negative** (predict legit, actually fraud): Bank loses money → worse outcome
- Banks tune the threshold based on their risk tolerance using the built-in slider.

### Why Three Models?
- **Logistic Regression**: Fast, interpretable, good baseline for regulatory explainability
- **Random Forest**: Best accuracy, handles non-linear fraud patterns, provides feature importances
- **Isolation Forest**: Catches novel fraud patterns without needing labeled data (unsupervised)

---

## 🔮 Future Improvements

- [ ] XGBoost / LightGBM for better gradient boosting performance
- [ ] SHAP values for per-prediction explainability
- [ ] Graph Neural Networks for transaction network analysis
- [ ] Online learning (river) for real-time model updates
- [ ] REST API backend (FastAPI) for production serving
- [ ] PostgreSQL integration for transaction storage
- [ ] Email/Slack alerts on fraud detection
- [ ] Model drift monitoring with Evidently AI

---

## 📝 License
MIT License — free for personal and commercial use.
