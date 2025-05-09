import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    return pd.DataFrame([{
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba)
    }])

results_df = pd.concat([
    evaluate_model("Random Forest", rf_model, X_test, y_test),
    evaluate_model("SVM", svm_model, X_test, y_test),
    evaluate_model("Neural Net", nn_model, X_test_scaled, y_test, is_nn=True)
], ignore_index=True)
results_df.to_csv("plots/model_evaluation_results.csv", index=False)

print(results_df)
# Assume results_df is already created as before
# Melt the DataFrame for easier plotting
melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

# Set the style
sns.set(style="whitegrid")

# Create a grouped bar plot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model")

# Add value labels on top of bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=9, color='black')

plt.title("📊 Performance des modèles")
plt.ylim(0, 1.05)
plt.legend(title="Model")
plt.tight_layout()

# Save and show
plt.savefig("plots/model_performance_comparison.png", dpi=300)
plt.show()





























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
    plt.savefig(f"plots/Matrice_de_Confusion_{name}.png")
    plt.show()
