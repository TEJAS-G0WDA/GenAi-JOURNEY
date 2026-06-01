import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================
# Load Dataset
# ==========================

data = load_breast_cancer()

X = data.data
y = data.target

# ==========================
# Train/Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Manual Calculations
# ==========================

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

accuracy = (TP + TN) / (TP + TN + FP + FN)

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1 = 2 * precision * recall / (precision + recall)

# ==========================
# Print Manual Results
# ==========================

print("===== MANUAL METRICS =====")

print(f"TP = {TP}")
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")

print(f"\nAccuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")

# ==========================
# Sklearn Results
# ==========================

print("\n===== SKLEARN METRICS =====")

print(f"Accuracy  = {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision = {precision_score(y_test, y_pred):.4f}")
print(f"Recall    = {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  = {f1_score(y_test, y_pred):.4f}")

def evaluate_range(metric_name, value):
    if value >= 0.90:
        status = "Excellent"
    elif value >= 0.80:
        status = "Good"
    elif value >= 0.70:
        status = "Acceptable"
    else:
        status = "Poor"

    print(
        f"{metric_name}: {value:.4f} | "
        f"Expected Range: 0.80 - 1.00 | "
        f"Status: {status}"
    )
    
print("\n===== PERFORMANCE REPORT =====")

evaluate_range("Accuracy", accuracy)
evaluate_range("Precision", precision)
evaluate_range("Recall", recall)
evaluate_range("F1 Score", f1)