import joblib
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.config import RAW_DATA_PATH

df = load_data(RAW_DATA_PATH)

X_train, X_test, y_train, y_test, scaler, target_encoder = preprocess_data(df)

joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(target_encoder, "artifacts/target_encoder.pkl")

joblib.dump(X_train, "artifacts/X_train.pkl")
joblib.dump(X_test, "artifacts/X_test.pkl")
joblib.dump(y_train, "artifacts/y_train.pkl")
joblib.dump(y_test, "artifacts/y_test.pkl")
