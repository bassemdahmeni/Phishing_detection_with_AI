import joblib
import torch
import torch.nn as nn
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
# Sklearn models
rf_model = joblib.load("models/random_forest_pipeline2.pkl")
svm_model = joblib.load("models/SVM_pipeline2.pkl")
X_test = pd.read_pickle("csv_files/X_test.pkl")
y_test = pd.read_pickle("csv_files/y_test.pkl")
scaler = load("models/scaler.pkl")
X_test_scaled = scaler.transform(X_test)
# PyTorch model (architecture + chargement)
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

# Charger le modèle
input_dim = 21
nn_model = PhishingNN(input_dim)
nn_model.load_state_dict(torch.load("models/phishing_nn.pth"))
nn_model.eval()
def evaluate_model(name, model, X, y_true, is_nn=False):
    if is_nn:
        model.eval()
        # Convertir X en tenseur PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            # Appeler directement le modèle pour obtenir les probabilités
            y_proba = model(X_tensor).squeeze().numpy()
        # Appliquer le seuil de 0.5 pour la classification binaire
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = model.decision_function(X)
        y_pred = (y_proba >= 0.5).astype(int)
    
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba)
    }

results = []
results.append(evaluate_model("Random Forest", rf_model, X_test, y_test))
results.append(evaluate_model("SVM", svm_model, X_test, y_test))
results.append(evaluate_model("Neural Net", nn_model, X_test_scaled, y_test, is_nn=True))
results_df = pd.DataFrame(results)
print(results_df)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

for name, model, X in [("Random Forest", rf_model, X_test),
                       ("SVM", svm_model, X_test),
                       ("Neural Net", nn_model, X_test_scaled)]:
    
    if name == "Neural Net":
        model.eval()
        # Convert X (already scaled) into a torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            # Get the model output and squeeze to remove extra dimensions if needed
            y_proba = model(X_tensor).squeeze().numpy()
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Phishing"])
    disp.plot()
    plt.title(f"Matrice de Confusion - {name}")
    plt.grid(False)
    plt.show()
