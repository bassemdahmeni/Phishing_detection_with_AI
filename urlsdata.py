import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les données finales
df = pd.read_csv("csv_files/final.csv")

# Supprimer les colonnes non utilisées
cols_to_drop = ["id", "url", "url_hash", "top_level_domain", "primary_domain", "created_at"]
df = df.drop(columns=cols_to_drop)

# Définir les colonnes de features
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

# Extraire X
X = df[FEATURE_COLUMNS]

# Appliquer une normalisation si besoin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Obtenir la dimension d'entrée
input_dim = X_scaled.shape[1]
print(f"input_dim = {input_dim}")
