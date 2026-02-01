# Load test data & model
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load artifacts
X_test = joblib.load("artifacts/X_test.pkl")
y_test = joblib.load("artifacts/y_test.pkl")
model = joblib.load("artifacts/models/final_model.pkl")

# Generate predictions
y_pred = model.predict(X_test)

# Core metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save metrics (IMPORTANT)
os.makedirs("artifacts/metrics", exist_ok=True)

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

joblib.dump(metrics, "artifacts/metrics/evaluation_metrics.pkl")
print("✅ Evaluation metrics saved")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
