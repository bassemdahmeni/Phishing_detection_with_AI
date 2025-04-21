import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_csv_files(file_paths):
    """Charge plusieurs fichiers CSV et les transforme en DataFrames."""
    print("[INFO] Chargement des fichiers CSV...")
    dataframes = []
    for file in file_paths:
        try:
            df = pd.read_csv(file, encoding='utf-8')
            dataframes.append(df)
            print(f"[SUCCESS] Chargé : {file} ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement de {file} : {e}")
    return dataframes

def clean_urls(df, url_column='url'):
    """Nettoie les URLs et enlève les valeurs manquantes."""
    print(f"[INFO] Nettoyage des URLs... ({df.shape[0]} entrées)")
    df = df.dropna(subset=[url_column])
    df[url_column] = df[url_column].str.strip()
    print(f"[SUCCESS] URLs nettoyées, {df.shape[0]} entrées restantes.")
    return df

def is_url_accessible(url):
    """Vérifie si l'URL est accessible (renvoie un statut 200)."""
    try:
        response = requests.get(url, timeout=7)
        return response.status_code == 200
    except requests.RequestException:
        return False

# def filter_accessible_urls(df, url_column='url'):
#     """Supprime les URLs non fonctionnelles du DataFrame."""
#     print(f"[INFO] Vérification de l'accessibilité des URLs... ({df.shape[0]} entrées)")
#     df['is_accessible'] = df[url_column].apply(is_url_accessible)
#     df = df[df['is_accessible']]
#     df.drop(columns=['is_accessible'], inplace=True)
#     print(f"[SUCCESS] URLs accessibles filtrées, {df.shape[0]} entrées restantes.")
#     return df
def filter_accessible_urls(df, url_column='url', max_workers=100):
    """Version parallélisée"""
    print(f"[INFO] Vérification de l'accessibilité des URLs... ({df.shape[0]} entrées)")
    urls = df[url_column].tolist()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(is_url_accessible, url): idx for idx, url in enumerate(urls)}
        results = [None] * len(urls)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    df['is_accessible'] = results
    df = df[df['is_accessible']]
    df.drop(columns=['is_accessible'], inplace=True)
    print(f"[SUCCESS] URLs accessibles filtrées, {df.shape[0]} entrées restantes.")
    return df


def extract_features(df, url_column='url'):
    """Extrait les features essentielles pour la détection de phishing."""
    print(f"[INFO] Extraction des features pour {df.shape[0]} URLs...")
    df['url_length'] = df[url_column].apply(len)
    df['contains_at'] = df[url_column].apply(lambda x: '@' in x)
    df['contains_dash'] = df[url_column].apply(lambda x: '-' in urlparse(x).netloc)
    df['num_subdomains'] = df[url_column].apply(lambda x: len(urlparse(x).netloc.split('.')) - 2)
    df['contains_ip'] = df[url_column].apply(lambda x: x.replace('.', '').isdigit())
    df['num_digits'] = df[url_column].apply(lambda x: sum(c.isdigit() for c in x))
    
    
    print(f"[SUCCESS] Features extraites avec succès !")
    return df

# def merge_dataframes(dataframes):
#     """Fusionne plusieurs DataFrames en un seul."""
#     print(f"[INFO] Fusion de {len(dataframes)} DataFrames...")
#     final_df = pd.concat(dataframes, ignore_index=True)
   
#     print(f"[SUCCESS] DataFrame final créé : {final_df.shape[0]} lignes, {final_df.shape[1]} colonnes")
#     return final_df
def merge_dataframes(dataframes):
    """Fusionne plusieurs DataFrames en un seul et ne conserve que les colonnes de features."""
    # Liste des colonnes à conserver (celles créées dans extract_features)
    feature_columns = [
        'url',               # La colonne originale d'URL (si vous voulez la garder)
        'url_length',        # Créée dans extract_features
        'contains_at',       # Créée dans extract_features
        'contains_dash',     # Créée dans extract_features
        'num_subdomains',    # Créée dans extract_features
        'contains_ip',       # Créée dans extract_features
        'num_digits',       # Créée dans extract_features
        'target'
    ]
    
    print(f"[INFO] Fusion de {len(dataframes)} DataFrames...")
    
    # Fusionner les DataFrames

    final_df = pd.concat(dataframes, ignore_index=True)
    
    
    # Ne conserver que les colonnes de features
    final_df = final_df[feature_columns]
    
    print(f"[SUCCESS] DataFrame final créé : {final_df.shape[0]} lignes, {final_df.shape[1]} colonnes")
    return final_df
def main_pipeline(file_paths):
    """Exécute l'ensemble du pipeline."""
    print("[START] Début du pipeline...")
    dfs = load_csv_files(file_paths)
    
    print("[STEP] Nettoyage des données...")
    dfs = [clean_urls(df) for df in dfs]
    
    print("[STEP] Vérification des URLs accessibles...")
    dfs = [filter_accessible_urls(df) for df in dfs]
    
    print("[STEP] Extraction des features...")
    dfs = [extract_features(df) for df in dfs]
    
    print("[STEP] Fusion des DataFrames...")
    final_df = merge_dataframes(dfs)
    
    print("[END] Pipeline terminé avec succès !")
    return final_df
