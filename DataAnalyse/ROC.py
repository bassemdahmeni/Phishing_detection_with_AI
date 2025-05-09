import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------------------------------
# Load your models and data
# -------------------------------------------------------
# Sklearn pipelines (Random Forest & SVM)
rf_model = joblib.load("models/random_forest_pipeline2.pkl")
svm_model = joblib.load("models/SVM_pipeline2.pkl")

# Data: load X_test and y_test (raw data for pipelines)
X_test = pd.read_pickle("csv_files/X_test.pkl")
y_test = pd.read_pickle("csv_files/y_test.pkl")

# Load your scaler and apply it to get scaled data for the PyTorch model
scaler = load("models/scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# Define and load the PyTorch model (Neural Net)
# -------------------------------------------------------
class PhishingNN(nn.Module):
    def __init__(self, input_size):
        super(PhishingNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Create an instance, note input_dim should match your feature count, e.g. 21
input_dim = 21
nn_model = PhishingNN(input_dim)
nn_model.load_state_dict(torch.load("models/phishing_nn.pth"))
nn_model.eval()

# -------------------------------------------------------
# Get ROC curve data for each model
# -------------------------------------------------------

plt.figure(figsize=(10, 7))

# 1. Random Forest (pipeline expects raw X_test)
if hasattr(rf_model, "predict_proba"):
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
else:
    # If not, fallback to decision_function and adjust threshold (usually threshold 0 for decision scores)
    y_proba_rf = rf_model.decision_function(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")

# 2. SVM (pipeline expects raw X_test)
if hasattr(svm_model, "predict_proba"):
    y_proba_svm = svm_model.predict_proba(X_test)[:, 1]
else:
    y_proba_svm = svm_model.decision_function(X_test)
    # Note: for decision_function, a threshold of 0 is standard—but roc_curve can handle raw scores
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_proba_svm)
auc_svm = roc_auc_score(y_test, y_proba_svm)
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})")

# 3. Neural Network (PyTorch model expects scaled data)
# Convert X_test_scaled to tensor
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    y_proba_nn = nn_model(X_tensor).squeeze().numpy()
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_proba_nn)
auc_nn = roc_auc_score(y_test, y_proba_nn)
plt.plot(fpr_nn, tpr_nn, label=f"Neural Net (AUC = {auc_nn:.2f})")

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', label="Random")

# Set plot properties
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("plots/ROC.png", dpi=300)
plt.show()
