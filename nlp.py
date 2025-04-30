import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Step 1: Data Collection
def collect_data(base_path):
    data = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    link = content.get('link', '')
                    text = content.get('content', '')
                    label = 'phishing' if 'phishing' in root else 'legitimate'
                    data.append({'link': link, 'content': text, 'label': label})
    return pd.DataFrame(data)

# Step 2: Data Preprocessing
def preprocess_data(df):
    df['content'] = df['content'].str.lower().str.replace(r'[^\w\s]', '')
    return df

# Step 3: Feature Extraction
def extract_features(df, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['content']).toarray()
    else:
        X = vectorizer.transform(df['content']).toarray()
    y = df['label']
    return X, y, vectorizer

# Step 4: Model Training
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Step 5: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, pos_label='phishing'))
    print("Recall:", recall_score(y_test, y_pred, pos_label='phishing'))
    print("F1-score:", f1_score(y_test, y_pred, pos_label='phishing'))
    print(classification_report(y_test, y_pred))

# Step 6: Save the Model and Vectorizer
def save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

# Step 7: Test the Model
def test_model(test_base_path, model_path, vectorizer_path):
    # Load the test data
    df_test = collect_data(test_base_path)
    df_test = preprocess_data(df_test)

    # Load the trained model and vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Extract features from the test data
    X_test, y_test, _ = extract_features(df_test, vectorizer)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

# Main function to train and save the model
def main_train(base_path, model_path, vectorizer_path):
    df = collect_data(base_path)
    df = preprocess_data(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)

    # Evaluate the model on the test set
    evaluate_model(model, X_test, y_test)

    # Save the model and vectorizer
    save_model_and_vectorizer(model, vectorizer, model_path, vectorizer_path)

# Replace 'your_base_path' with the path to your training folders
# Replace 'phishing_model.pkl' and 'vectorizer.pkl' with your desired file paths
main_train(r"C:\Users\user\OneDrive\Bureau\Academics\pcd\extract\links_train", 'phishing_model.pkl', 'vectorizer.pkl')

# Replace 'your_test_base_path' with the path to your test folders
test_model(r"C:\Users\user\OneDrive\Bureau\Academics\pcd\extract\test_links", 'phishing_model.pkl', 'vectorizer.pkl')
















