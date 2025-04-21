import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Liste des noms de colonnes utilisés comme features (ordre important)
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

def main():
    # 1. Charger le CSV et supprimer les colonnes inutiles
    df = pd.read_csv("csv_files/final.csv")
    df = df.drop(columns=["id", "url", "url_hash", "top_level_domain", "primary_domain", "created_at"])
    # 2. Sélectionner les features et la target en conservant l'ordre des colonnes
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    # 3. Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 4. Créer un pipeline avec StandardScaler et RandomForestClassifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("🌲 Début de l'entraînement du modèle Random Forest avec pipeline...")
    pipeline.fit(X_train, y_train)
    print("✅ Entraînement terminé.")
    
    # 5. Évaluer le modèle sur l'ensemble de test
    y_pred = pipeline.predict(X_test)
    
    print("\n📊 Évaluation du modèle :")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # 6. Sauvegarder le pipeline (incluant le scaler et le modèle)
    joblib.dump(pipeline, "models/random_forest_pipeline2.pkl")
    print("\n💾 Pipeline sauvegardé sous 'random_forest_pipeline.pkl'.")
    scaler = pipeline.named_steps['scaler']
    joblib.dump(scaler, "models/scaler.pkl")
    print("💾 Scaler sauvegardé séparément sous 'scaler.pkl'.")
    

if __name__ == '__main__':
    main()
