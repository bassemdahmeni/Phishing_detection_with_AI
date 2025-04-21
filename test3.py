import joblib
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse

# === 1. Liste des features utilisées lors de l'entraînement ===
FEATURE_COLUMNS = [
    'url_length', 'num_special_chars', 'digit_to_letter_ratio',
    'contains_ip', 'primary_domain_length', 'num_digits_primary_domain',
    'num_non_alphanumeric_primary', 'num_hyphens_primary', 'num_ats_primary',
    'num_dots_subdomain', 'num_subdomains', 'num_double_slash',
    'num_subdirectories', 'contains_encoded_space', 'uppercase_dirs',
    'single_char_dirs', 'num_special_chars_path', 'num_zeroes_path',
    'uppercase_ratio', 'params_length', 'num_queries'
]

# === 2. Fonction pour extraire les features d'une URL ===
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

# === 3. Charger le modèle entraîné (avec noms de colonnes) ===
rf_model = joblib.load("random_forest_pipeline2.pkl")
print("[✅] Modèle chargé avec succès.")

# === 4. Fonction de prédiction propre ===
def predict_url(url):
    features_df = extract_features_from_url(url)
    prediction = rf_model.predict(features_df)[0]
    return "Phishing" if prediction == 1 else "Safe"

# === 5. URLs à tester ===
urls_to_test = [
   "https://youtube.com",
   "http://p3.zbjimg.com/task/2009-06/06/98428/07c9mfhe.zip"
   "https://share.hsforms.com/1JWlO6MOnQna9WurFUuBnywtei4v",
   "http://app.dialoginsight.com",
   "https://www.todayshomeowner.com"


]

print("\n🔍 Résultats des prédictions :")
for url in urls_to_test:
    result = predict_url(url)
    print(f"URL: {url} --> {result}")
