import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# === Load Models and Vectorizer ===
rf_model = joblib.load("models/random_forest_pipeline2.pkl")
print("[✅] Random Forest model loaded.")

nlp_model = joblib.load('models/phishing_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
print("[✅] NLP model loaded.\n")

# === Feature Extraction for Random Forest ===
FEATURE_COLUMNS = [
    'url_length', 'num_special_chars', 'digit_to_letter_ratio',
    'contains_ip', 'primary_domain_length', 'num_digits_primary_domain',
    'num_non_alphanumeric_primary', 'num_hyphens_primary', 'num_ats_primary',
    'num_dots_subdomain', 'num_subdomains', 'num_double_slash',
    'num_subdirectories', 'contains_encoded_space', 'uppercase_dirs',
    'single_char_dirs', 'num_special_chars_path', 'num_zeroes_path',
    'uppercase_ratio', 'params_length', 'num_queries'
]

def extract_features_from_url(url):
    parsed_url = urlparse(url.strip().lower())
    primary_domain = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path
    query = parsed_url.query

    features = {
        'url_length': len(url),
        'num_special_chars': len(re.findall(r'[;_?=&]', url)),
        'digit_to_letter_ratio': sum(c.isdigit() for c in url) / (sum(c.isalpha() for c in url) + 1),
        'contains_ip': int(primary_domain.replace('.', '').isdigit()),
        'primary_domain_length': len(primary_domain),
        'num_digits_primary_domain': sum(c.isdigit() for c in primary_domain),
        'num_non_alphanumeric_primary': sum(not c.isalnum() for c in primary_domain),
        'num_hyphens_primary': primary_domain.count('-'),
        'num_ats_primary': primary_domain.count('@'),
        'num_dots_subdomain': parsed_url.netloc.count('.'),
        'num_subdomains': len(parsed_url.netloc.split('.')) - 2,
        'num_double_slash': path.count('//'),
        'num_subdirectories': path.count('/'),
        'contains_encoded_space': int('%20' in path),
        'uppercase_dirs': int(any(c.isupper() for c in path)),
        'single_char_dirs': int(any(len(part) == 1 for part in path.split('/'))),
        'num_special_chars_path': len(re.findall(r'[@_&=]', path)),
        'num_zeroes_path': path.count('0'),
        'uppercase_ratio': sum(c.isupper() for c in path) / (len(path) + 1),
        'params_length': len(query),
        'num_queries': query.count('&') + (1 if query else 0)
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)

# === NLP Model Feature Extraction ===
def fetch_and_preprocess_url(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ')
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text
    except Exception as e:
        print(f"[⚠️] Error fetching content from {url}: {e}")
        return ""

def predict_combined(url, w_lexical=0.5, w_nlp=0.5, threshold=0.5):
    # Lexical prediction
    lexical_features = extract_features_from_url(url)
    prob_lexical = rf_model.predict_proba(lexical_features)[0][1]

    # NLP prediction (try to fetch content)
    page_text = fetch_and_preprocess_url(url)
    if not page_text.strip():  # Fallback condition
        print(f"\n[⚠️] No content available for NLP. Falling back to Random Forest only for: {url}")
        print(f"  - Lexical Prob: {prob_lexical:.2f}")
        label = "Phishing" if prob_lexical >= threshold else "Safe"
        print(f"  - Final Label : {label} (Lexical only)")
        return label

    # NLP probability
    vectorized_text = vectorizer.transform([page_text]).toarray()
    prob_nlp = nlp_model.predict_proba(vectorized_text)[0][1]

    # Combined late fusion score
    final_score = w_lexical * prob_lexical + w_nlp * prob_nlp
    label = "Phishing" if final_score >= threshold else "Safe"

    print(f"\n🔎 URL: {url}")
    print(f"  - Lexical Prob: {prob_lexical:.2f}")
    print(f"  - NLP Prob    : {prob_nlp:.2f}")
    print(f"  - Final Score : {final_score:.2f} --> {label}")

    return label



urls_to_test = [
    "http://vtv16.com",
    "https://facebook.com",
    "https://www.ar.aliexpress.com"
]

for url in urls_to_test:
    predict_combined(url, w_lexical=0.4, w_nlp=0.6)
