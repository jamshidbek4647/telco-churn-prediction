# Telco Customer Churn Prediction

### ✨ Features
- Best model -- **84% ROC-AUC** performance with Logistic Regression (CV: 84.47%)
- Next best model (slight difference) -- **84% ROC-AUC** performance with Gradient Boosting
- **4 ML Models** compared: Logistic Regression, Random Forest, Gradient Boosting, Neural Network
- **3 Analysis Modes**: Individual, Batch, Performance Evaluation
- **Fixed UI** with perfect color contrast
- **Relative Paths** works from any directory

### 📁 Project Structure

```
├── data/
│   └── telco-churn.csv          # Real Kaggle dataset
├── ├── models/
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
| **Gradient Boosting** | **0.8417** | **0.5765** | **0.7956** |
| Logistic Regression | 0.8415 | 0.6168 | 0.7381 |
| Random Forest | 0.8376 | 0.6280 | 0.7722 |
| Neural Network | 0.8348 | 0.5421 | 0.7878 |

### 🎨 Fixed Issues

✅ **Color Contrast**: Dark text on light backgrounds (perfectly readable)
✅ **Relative Paths**: Uses config.py, works from any directory  
✅ **Real Dataset**: Kaggle Telco Customer Churn data
✅ **Performance**: 84% ROC-AUC vs 70% with synthetic data

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

**Training:**
- 80/20 stratified split
- 5-fold cross-validation
- Class weights for imbalance
- StandardScaler for numerical features

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

**Batch Analysis:**
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

---

**Built for academic excellence** | February 2026