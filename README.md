```markdown
# Telco Customer Churn Prediction

### ✨ Features
- **84.57% ROC-AUC** performance with Gradient Boosting (after hyperparameter tuning)
- **4 fodels** compared: Logistic Regression, Random Forest, Gradient Boosting, Neural Network
- **3 Analysis Modes**: Individual Analysis, Bulk Customer Analysis, Model Performance
- **Fixed UI** with perfect color contrast
- **Relative Paths** works from any directory

### 📁 Project Structure

```
├── data/
│   └── telco-churn.csv          # Real Kaggle dataset
├── models/
│   ├── models.pkl               # Trained models
│   ├── scaler.pkl               # Feature scaler
│   ├── ordinal_encoder.pkl      # Ordinal encoder for Tenure_Category
│   ├── feature_names.pkl        # Feature names
│   ├── model_metrics.json       # Performance metrics
│   └── test_data.pkl            # Test set
├── config.py                    # Configuration
├── app.py                       # Streamlit app
├── train_models.py              # Training pipeline
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

### 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models
python train_models.py

# 3. Run app
streamlit run app.py
```

### 📊 Model Performance

| Model | ROC-AUC | F1-Score | Accuracy |
|-------|---------|----------|----------|
| **Gradient Boosting** | **0.8457** | **0.5771** | **0.8034** |
| Logistic Regression | 0.8413 | 0.6180 | 0.7402 |
| Random Forest | 0.8407 | 0.6323 | 0.7672 |
| Neural Network | 0.8354 | 0.5810 | 0.7963 |


### 🔧 Configuration

Edit `config.py` to customize:

```python
# Risk thresholds
LOW_RISK_THRESHOLD = 0.30   # <30% = Low
HIGH_RISK_THRESHOLD = 0.70  # >70% = High

# Colors
COLORS = {
    'low_risk_text': '#064E3B',     # Dark green
    'medium_risk_text': '#78350F',  # Dark orange
    'high_risk_text': '#7F1D1D',    # Dark red
}
```

### 📚 Methodology

**Feature Engineering:**
1. Tenure Category (New/Medium/Long)
2. Total Services (count)
3. Average Monthly Spend
4. Contract Value
5. Service Density

**Encoding:**
- Ordinal Encoding for Tenure_Category (preserves New → Medium → Long order)
- One-Hot Encoding for remaining categorical features

**Training:**
- 80/20 stratified split
- 5-fold cross-validation
- Class weights for imbalance
- StandardScaler for numerical features
- RandomizedSearchCV hyperparameter tuning (n_iter=100, cv=5) for Gradient Boosting and Random Forest

**Evaluation:**
- ROC-AUC (primary metric)
- Precision-Recall curves
- Confusion matrices
- F1-Score, Precision, Recall

### 🎯 Usage

**Individual Analysis:**
1. Enter customer details
2. Click "Predict"
3. View risk level and top factors

**Bulk Customer Analysis:**
1. Upload CSV/XLSX
2. Click "Analyze"
3. Download results

**Model Performance:**
- Compare all 4 models
- View metrics and curves
- Select different models

### 🙏 Acknowledgments

- Dataset: IBM Telco Customer Churn (Kaggle)
- Framework: Streamlit
- ML Library: Scikit-learn

### 🔗 Source Code
Available at: https://github.com/jamshidbek4647/telco-churn-prediction

---

**Built for academic excellence** | February 2026
```
