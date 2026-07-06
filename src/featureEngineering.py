"""
feature_engineering.py
Project: Health Insurance Cross Sell Prediction
Contains all feature engineering functions.
"""

import numpy as np
import pandas as pd
from src.config import FEATURE_COLUMNS, TRAIN_DATA_PATH
from src.preprocessing import preprocess_training_data


# ==========================================================
# PREMIUM PER AGE
# ==========================================================
def premium_per_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual Premium / Age
    """
    df["Premium_Per_Age"] = (df["Annual_Premium"] /df["Age"].replace(0, 1))
    return df

# ==========================================================
# CUSTOMER TENURE (YEARS)
# ==========================================================
def vintage_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert days into years.
    """
    df["Vintage_Years"] = (
        df["Vintage"] / 365
    ).round(2)
    return df

# ==========================================================
# HIGH PREMIUM FLAG
# ==========================================================
def high_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customers paying premium above 75 percentile.
    """
    threshold = df["Annual_Premium"].quantile(0.75)
    df["High_Premium"] = (
        df["Annual_Premium"] > threshold
    ).astype(int)
    return df

# ==========================================================
# YOUNG DRIVER
# ==========================================================
def young_driver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age < 30
    """
    df["Young_Driver"] = (
        df["Age"] < 30
    ).astype(int)
    return df

# ==========================================================
# SENIOR CITIZEN
# ==========================================================
def senior_citizen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age >= 60
    """
    df["Senior_Citizen"] = (
        df["Age"] >= 60
    ).astype(int)
    return df

# ==========================================================
# HAS VEHICLE DAMAGE & NOT INSURED
# ==========================================================
def damage_not_insured(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strong business feature.
    Vehicle damaged before AND
    Not previously insured
    """
    df["Damage_Not_Insured"] = (
        (df["Vehicle_Damage"] == 1) &
        (df["Previously_Insured"] == 0)
    ).astype(int)
    return df

# ==========================================================
# PREMIUM CATEGORY
# ==========================================================
def premium_category(df: pd.DataFrame) -> pd.DataFrame:
    df["Premium_Category"] = pd.qcut(
        df["Annual_Premium"],
        q=4,
        labels=[
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )
    premium_map = {
        "Low":0,
        "Medium":1,
        "High":2,
        "Very High":3
    }
    df["Premium_Category"] = (
        df["Premium_Category"]
        .map(premium_map)
        .astype(int)
    )
    return df

# ==========================================================
# AGE x VEHICLE AGE
# ==========================================================
def age_vehicle_interaction(df: pd.DataFrame):
    """
    Interaction feature
    """
    df["Age_VehicleAge"] = ( df["Age"] * df["Vehicle_Age"])
    return df

# ==========================================================
# PREMIUM x VINTAGE
# ==========================================================
def premium_vintage(df: pd.DataFrame):
    df["Premium_Vintage"] = ( df["Annual_Premium"]*df["Vintage"])
    return df

# ==========================================================
# LOG PREMIUM
# ==========================================================
def log_premium(df):
    df["Log_Premium"] = np.log1p(df["Annual_Premium"])
    return df

# ==========================================================
# SELECT FEATURES
# ==========================================================

def select_features(df):
    features = FEATURE_COLUMNS
    return df[features]

# ==========================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ==========================================================
def feature_engineering(df):
    df = premium_per_age(df)
    df = vintage_years(df)
    df = high_premium(df)
    df = young_driver(df)
    df = senior_citizen(df)
    df = damage_not_insured(df)
    df = premium_category(df)
    df = age_vehicle_interaction(df)
    df = premium_vintage(df)
    df = log_premium(df)
    return df

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(TRAIN_DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_training_data(df)
    X_train = feature_engineering(X_train)
    print(X_train.head())
    print(X_train.columns)