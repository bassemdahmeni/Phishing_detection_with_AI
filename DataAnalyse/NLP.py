import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import random

# Load model and vectorizer
model = joblib.load('models/phishing_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Function to fetch and preprocess URL content
def fetch_and_preprocess_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ')
        text = text.lower().replace(r'[^\w\s]', '')
        return text
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

# Load dataset
df = pd.read_csv('csv_files/final.csv')

# Random sample of URLs
sampled_df = df.sample(n=30, random_state=42)  # You can change the number

# Store true labels and predictions
true_labels = []
predicted_probs = []

for _, row in sampled_df.iterrows():
    url = row['url']
    label = row['target']  # Assuming 'target' column has 0 or 1
    text = fetch_and_preprocess_url(url)
    if text:
        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]  # Probability of phishing
        true_labels.append(label)
        predicted_probs.append(prob)
    else:
        print(f"Skipping URL due to fetch error: {url}")

# Convert probabilities to binary predictions
predicted_labels = [1 if prob >= 0.5 else 0 for prob in predicted_probs]

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Phishing'])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Phishing BERT Model').plot()
plt.title("ROC Curve")
plt.show()
