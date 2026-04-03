"""
Configuration file for Telco Churn Prediction
Centralized path and settings management
"""

import os

# Get the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "telco-churn.csv")

# ============================================================================
# MODEL PATHS
# ============================================================================
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_PATH = os.path.join(MODELS_DIR, "models.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
TEST_DATA_PATH = os.path.join(MODELS_DIR, "test_data.pkl")
ORDINAL_ENCODER_PATH = os.path.join(MODELS_DIR, "ordinal_encoder.pkl")
TARGET_ENCODER_PATH = os.path.join(MODELS_DIR, "target_encoder.pkl")

# ============================================================================
# MODEL SETTINGS
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================================
# RISK THRESHOLDS
# ============================================================================
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.70

# ============================================================================
# UI COLOR SCHEME
# ============================================================================
COLORS = {
    'low_risk': '#10B981',      # Green
    'medium_risk': '#F59E0B',   # Orange
    'high_risk': '#EF4444',     # Red
    'primary': '#3B82F6',       # Blue
    'background': '#1F2937',    # Dark gray
    'text': '#F9FAFB',          # Light gray
    
    # Text colors for contrast (dark on light backgrounds)
    'low_risk_text': '#064E3B',     # Dark green
    'medium_risk_text': '#78350F',  # Dark orange/brown
    'high_risk_text': '#7F1D1D',    # Dark red
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def get_dataset_path():
    """
    Returns the path to the dataset
    Raises FileNotFoundError if dataset doesn't exist
    """
    if os.path.exists(DATASET_PATH):
        return DATASET_PATH
    else:
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}\n"
            f"Please ensure 'telco-churn.csv' is in the 'data' folder."
        )

# ============================================================================
# INITIALIZATION
# ============================================================================
# Create directories on import
ensure_directories()
