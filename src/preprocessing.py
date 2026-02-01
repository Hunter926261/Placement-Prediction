import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


# Split features & target
def split_features_target(df: pd.DataFrame):
    X = df.drop(
        columns=[
            TARGET_COLUMN,
            "StudentID",
            "ExtracurricularActivities",
            "PlacementTraining"
        ]
    )
    y = df[TARGET_COLUMN]
    return X, y


# Encode the target variable
def encode_target(y: pd.Series):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


# Identify numerical & categorical features
def identify_feature_types(X: pd.DataFrame):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    return numerical_cols, categorical_cols


# Train–Test Split (NO leakage)
def split_train_test(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# Scale numerical features
def scale_numerical_features(X_train, X_test, numerical_cols):
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train_scaled, X_test_scaled, scaler


# Encode categorical features
def encode_categorical_features(X_train, X_test, categorical_cols):
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    X_train_encoded, X_test_encoded = X_train_encoded.align(
        X_test_encoded, join='left', axis=1, fill_value=0
    )

    return X_train_encoded, X_test_encoded


# Final preprocessing function (ORCHESTRATOR)
def preprocess_data(df: pd.DataFrame):
    X, y = split_features_target(df)
    y_encoded, target_encoder = encode_target(y)

    num_cols, cat_cols = identify_feature_types(X)

    X_train, X_test, y_train, y_test = split_train_test(X, y_encoded)

    X_train, X_test, scaler = scale_numerical_features(
        X_train, X_test, num_cols
    )

    X_train, X_test = encode_categorical_features(
        X_train, X_test, cat_cols
    )

    return X_train, X_test, y_train, y_test, scaler, target_encoder


