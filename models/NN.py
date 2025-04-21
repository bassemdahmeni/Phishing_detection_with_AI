import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Config
FEATURE_COLUMNS = [
    'url_length', 'num_special_chars', 'digit_to_letter_ratio', 'contains_ip',
    'primary_domain_length', 'num_digits_primary_domain', 'num_non_alphanumeric_primary',
    'num_hyphens_primary', 'num_ats_primary', 'num_dots_subdomain', 'num_subdomains',
    'num_double_slash', 'num_subdirectories', 'contains_encoded_space', 'uppercase_dirs',
    'single_char_dirs', 'num_special_chars_path', 'num_zeroes_path', 'uppercase_ratio',
    'params_length', 'num_queries'
]
TARGET_COLUMN = "target"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

# Define neural network
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

def main():
    # 1. Load and clean data
    df = pd.read_csv("csv_files/final.csv")
    df = df.drop(columns=["id", "url", "url_hash", "top_level_domain", "primary_domain", "created_at"])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # 5. Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 6. Model, loss, optimizer
    model = PhishingNN(input_size=X_train_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training loop
    print("🧠 Début de l'entraînement du réseau de neurones...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    # 8. Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred_labels = (y_pred >= 0.5).int()

    print("\n📊 Évaluation du modèle :")
    print("Accuracy:", accuracy_score(y_test_tensor, y_pred_labels))
    print("Classification Report:\n", classification_report(y_test_tensor, y_pred_labels))
    print("Confusion Matrix:\n", confusion_matrix(y_test_tensor, y_pred_labels))

    # 9. Sauvegarde du modèle PyTorch
    torch.save(model.state_dict(), "models/phishing_nn.pth")
    print("\n💾 Modèle sauvegardé sous 'phishing_nn.pth'.")

if __name__ == "__main__":
    main()
