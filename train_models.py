"""
Telco Churn Prediction - Model Training Pipeline
(Uses Kaggle Telco dataset for Churn Analysis)
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import json
import os

# Import config
import config

print("="*80)
print("TELCO CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================
print("\n[1/8] Loading dataset...")
try:
    dataset_path = config.get_dataset_path()
    print(f"   Dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)
    print(f"   Shape: {df.shape}")
    print(f"   Churn distribution:")
    print(f"   {df['Churn'].value_counts(normalize=True)}")
except FileNotFoundError as e:
    print(f"   ERROR: {e}")
    exit(1)

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n[2/8] Preprocessing data...")

# Handle TotalCharges (has spaces that need converting)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_total_charges = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
print(f"   Filled {df['TotalCharges'].isna().sum()} missing TotalCharges values")

# Drop customerID
df_processed = df.drop('customerID', axis=1)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n[3/8] Engineering features...")

# Feature 1: Tenure Category
def categorize_tenure(tenure):
    if tenure <= 12:
        return 'New'
    elif tenure <= 36:
        return 'Medium'
    else:
        return 'Long'

df_processed['Tenure_Category'] = df_processed['tenure'].apply(categorize_tenure)

# Feature 2: Total Services
service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_processed['TotalServices'] = 0
for col in service_cols:
    df_processed['TotalServices'] += (df_processed[col].isin(['Yes', 'DSL', 'Fiber optic'])).astype(int)

# Feature 3: Average Monthly Spend
df_processed['AvgMonthlySpend'] = df_processed.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'], 
    axis=1
)

# Feature 4: Contract Value
contract_months = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df_processed['ContractValue'] = df_processed.apply(
    lambda row: row['MonthlyCharges'] * contract_months.get(row['Contract'], 1), 
    axis=1
)

# Feature 5: Service Density
df_processed['ServiceDensity'] = df_processed.apply(
    lambda row: row['TotalServices'] / row['MonthlyCharges'] if row['MonthlyCharges'] > 0 else 0,
    axis=1
)

print(f"   Engineered 5 features")
print(f"   New shape: {df_processed.shape}")


# ============================================================================
# 4. ENCODING CATEGORICAL VARIABLES
# (Ordinal → Tenure_Category | Target → PaymentMethod, Contract | One-Hot → rest)
# ============================================================================
print("\n[4/8] Encoding categorical variables...")

X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()



print(f"   Categorical: {len(categorical_cols)}")
print(f"   Numerical: {len(numerical_cols)}")

# Ordinal encoding for Tenure_Category (has natural order)
ordinal_encoder = OrdinalEncoder(categories=[['New', 'Medium', 'Long']])
X['Tenure_Category'] = ordinal_encoder.fit_transform(X[['Tenure_Category']])

# One-hot encode remaining nominal categoricals
nominal_cols = [col for col in categorical_cols 
                if col not in ['Tenure_Category']]
X_encoded = pd.get_dummies(X, columns=nominal_cols)

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================
print("\n[5/8] Splitting data (stratified 80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=config.TEST_SIZE, 
    random_state=config.RANDOM_STATE, 
    stratify=y
)

print(f"   Training: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")
print(f"   Train churn rate: {y_train.mean():.2%}")
print(f"   Test churn rate: {y_test.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# ============================================================================
# 6. TRAIN MODELS
# ============================================================================
print("\n[6/8] Training models...")

# Hyperparameter grids
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

# Tune Random Forest
print("\n   Tuning Random Forest...")
rf_base = RandomForestClassifier(
    random_state=config.RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1
)
rf_search = RandomizedSearchCV(
    rf_base, rf_param_grid,
    n_iter=100, cv=5,
    scoring='roc_auc',
    random_state=config.RANDOM_STATE,
    n_jobs=-1, verbose=1
)
rf_search.fit(X_train, y_train)
print(f"   Best RF params: {rf_search.best_params_}")
print(f"   Best RF CV ROC-AUC: {rf_search.best_score_:.4f}")

# Tune Gradient Boosting
print("\n   Tuning Gradient Boosting...")
gb_base = GradientBoostingClassifier(
    random_state=config.RANDOM_STATE
)
gb_search = RandomizedSearchCV(
    gb_base, gb_param_grid,
    n_iter=100, cv=5,
    scoring='roc_auc',
    random_state=config.RANDOM_STATE,
    n_jobs=-1, verbose=1
)
gb_search.fit(X_train, y_train)
print(f"   Best GB params: {gb_search.best_params_}")
print(f"   Best GB CV ROC-AUC: {gb_search.best_score_:.4f}")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',
        solver='lbfgs'
    ),
    'Random Forest': rf_search.best_estimator_,      
    'Gradient Boosting': gb_search.best_estimator_,  
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=config.RANDOM_STATE,
        early_stopping=True
    )
}
trained_models = {}
model_metrics = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Use scaled data for Neural Network and Logistic Regression
    if name in ['Neural Network', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store metrics
    model_metrics[name] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred.tolist(),
        'probabilities': y_pred_proba.tolist()
    }
    
    # Store model
    trained_models[name] = model
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      ROC-AUC: {roc_auc:.4f}")
    print(f"      F1-Score: {f1:.4f}")

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n[7/8] Cross-validation on best model...")

best_model_name = max(model_metrics.items(), key=lambda x: x[1]['roc_auc'])[0]
best_model = trained_models[best_model_name]

print(f"   Best model: {best_model_name}")

cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

if best_model_name in ['Neural Network', 'Logistic Regression']:
    X_for_cv = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)
else:
    X_for_cv = X_encoded

cv_scores = cross_val_score(best_model, X_for_cv, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================================
# 8. SAVE MODELS AND ARTIFACTS
# ============================================================================
print("\n[8/8] Saving models and artifacts...")

config.ensure_directories()

# Save models
with open(config.MODELS_PATH, 'wb') as f:
    pickle.dump(trained_models, f)

# Save scaler
with open(config.SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# Save ordinal encoder 
with open(config.ORDINAL_ENCODER_PATH, 'wb') as f:
    pickle.dump(ordinal_encoder, f)

# Save feature names
with open(config.FEATURE_NAMES_PATH, 'wb') as f:
    pickle.dump(X_encoded.columns.tolist(), f)

# Save metrics
with open(config.METRICS_PATH, 'w') as f:
    json.dump(model_metrics, f, indent=2)

# Save test data
test_data = {
    'X_test': X_test.values.tolist(),
    'X_test_scaled': X_test_scaled.values.tolist(),
    'y_test': y_test.tolist(),
    'feature_names': X_encoded.columns.tolist()
}
with open(config.TEST_DATA_PATH, 'wb') as f:
    pickle.dump(test_data, f)

print(f"\n   ✓ Saved to {config.MODELS_DIR}/")
print("   ✓ models.pkl")
print("   ✓ scaler.pkl")
print("   ✓ feature_names.pkl")
print("   ✓ model_metrics.json")
print("   ✓ test_data.pkl")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"ROC-AUC: {model_metrics[best_model_name]['roc_auc']:.4f}")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nAll models ready for deployment!")
print("Run: streamlit run app.py")
print("="*80)
