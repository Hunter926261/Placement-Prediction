import os
import joblib
from sklearn.linear_model import LogisticRegression


# Load processed artifacts
X_train = joblib.load("artifacts/X_train.pkl")
y_train = joblib.load("artifacts/y_train.pkl")


# Ensure model directory exists
os.makedirs("artifacts/models", exist_ok=True)

# Train final model
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train, y_train)

# Save model
joblib.dump(final_model, "artifacts/models/final_model.pkl")

print("✅ Final Logistic Regression model saved")
