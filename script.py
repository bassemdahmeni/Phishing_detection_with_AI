import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
FEATURE_COLUMNS = [
    'url_length',
    'num_special_chars',
    'digit_to_letter_ratio',
    'contains_ip',
    'primary_domain_length',
    'num_digits_primary_domain',
    'num_non_alphanumeric_primary',
    'num_hyphens_primary',
    'num_ats_primary',
    'num_dots_subdomain',
    'num_subdomains',
    'num_double_slash',
    'num_subdirectories',
    'contains_encoded_space',
    'uppercase_dirs',
    'single_char_dirs',
    'num_special_chars_path',
    'num_zeroes_path',
    'uppercase_ratio',
    'params_length',
    'num_queries'
]

TARGET_COLUMN = "target"  # Nom de la colonne cible
df = pd.read_csv("csv_files/final.csv")
df = df.drop(columns=["id", "url", "url_hash", "top_level_domain", "primary_domain", "created_at"])
# 2. Sélectionner les features et la target en conservant l'ordre des colonnes
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]
# 3. Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_test.to_pickle("csv_files/X_test.pkl")
y_test.to_pickle("csv_files/y_test.pkl")

print("X_test et y_test ont été sauvegardés avec succès.")