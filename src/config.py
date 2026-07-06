"""
config.py

Central configuration file for the
Health Insurance Cross Sell Prediction project.
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(f"Project Root: {PROJECT_ROOT}")

# ==========================================================
# DATA PATHS
# ==========================================================
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"

# ==========================================================
# MODEL DIRECTORY
# ==========================================================
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
SHAP_EXPLAINER_PATH = MODEL_DIR / "shap_explainer.pkl"

# ==========================================================
# LOG DIRECTORY
# ==========================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRAIN_LOG = LOG_DIR / "training.log"
PREDICTION_LOG = LOG_DIR / "prediction.log"
API_LOG = LOG_DIR / "api.log"

# ==========================================================
# TARGET COLUMN
# ==========================================================
TARGET = "Response"

# ==========================================================
# RANDOM STATE
# ==========================================================
RANDOM_STATE = 42

# ==========================================================
# TRAIN TEST SPLIT
# 
# ==========================================================
TEST_SIZE = 0.10

# ==========================================================
# CROSS VALIDATION
# ==========================================================
N_SPLITS = 5

# ==========================================================
# XGBOOST DEFAULT PARAMETERS
# ==========================================================
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ==========================================================
# OPTUNA SETTINGS
# ==========================================================
OPTUNA_TRIALS = 50

# ==========================================================
# SHAP SETTINGS
# ==========================================================
BACKGROUND_SAMPLES = 1000

# ==========================================================
# API SETTINGS
# ==========================================================
API_TITLE = "Health Insurance Cross Sell API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "Predict whether a customer is likely "
    "to purchase vehicle insurance."
)

# ==========================================================
# STREAMLIT SETTINGS
# ==========================================================
PAGE_TITLE = "Health Insurance Cross Sell"
PAGE_ICON = "🚗"
LAYOUT = "wide"

# ==========================================================
# FEATURE LIST
# ==========================================================
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Driving_License",
    "Region_Code",
    "Previously_Insured",
    "Vehicle_Age",
    "Vehicle_Damage",
    "Annual_Premium",
    "Policy_Sales_Channel",
    "Vintage",
    "Age_Group",
    "Premium_Per_Age",
    "Vintage_Years",
    "High_Premium",
    "Young_Driver",
    "Senior_Citizen",
    "Damage_Not_Insured",
    "Premium_Category",
    "Age_VehicleAge",
    "Premium_Vintage",
    "Log_Premium"
]