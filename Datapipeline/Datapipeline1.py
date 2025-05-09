import pandas as pd
import re
import time
import hashlib
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from Phishing_detection_with_AI.Datapipeline.Database_connection import supabase  # Importer la connexion Supabase

# 1️⃣ Charger les CSVs directement dans Supabase
import pandas as pd
import pandas as pd
import hashlib
def get_existing_urls(urls):
    """Récupère les URLs existantes en batch pour éviter les requêtes trop longues"""
    existing_urls = set()
    batch_size = 200  # Ajuste en fonction des limites
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        try:
            response = supabase.table("raw_urls") \
                .select("url") \
                .in_("url", batch) \
                .execute()
            if response.data:
                existing_urls.update(row["url"].strip().lower() for row in response.data)
        except Exception as e:
            print(f"[WARNING] Problème avec le batch {i//batch_size + 1}: {e}")

    return existing_urls

def load_csv_to_supabase(file_paths):
    """Charge CSVs en vérifiant les doublons sans insérer url_hash"""
    print("[INFO] Début du chargement optimisé...")
    
    for file in file_paths:
        try:
            # 1. Charger et nettoyer les données
            df = pd.read_csv(file, encoding='utf-8')[['url', 'target']]
            df['url'] = df['url'].str.strip().str.lower()
            df = df[df['url'].notna()]
            df = df[df['url'] != '']
            df = df.drop_duplicates(subset=['url'], keep='first')
            df = df[df['url'].apply(lambda x: len(x.encode('utf-8')) <= 8191)]

            # 2. Calculer les hashs juste pour la vérification
            df['temp_hash'] = df['url'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
            
            # 3. Vérifier les URLs existantes par leur URL (pas par hash)
            existing_urls = get_existing_urls(df['url'].tolist())

            # 4. Filtrer les nouvelles URLs
            new_data = df[~df["url"].isin(existing_urls)]
            
            if new_data.empty:
                print(f"[INFO] Aucune nouvelle URL dans {file}")
                continue

            # 5. Insérer seulement les colonnes non-générées
            batch_size = 500
            total = len(new_data)
            
            for i in range(0, total, batch_size):
                batch = new_data.iloc[i:i+batch_size]
                records = batch[['url', 'target']].to_dict('records')
                
                try:
                    supabase.table("raw_urls").insert(records).execute()
                    print(f"[PROGRESS] Lot {i//batch_size + 1}: {min(i+batch_size, total)}/{total}")
                except Exception as e:
                    print(f"[WARNING] Erreur sur le lot {i//batch_size + 1}: {e}")
                    # Réessayer avec des URLs individuelles
                    for _, row in batch.iterrows():
                        try:
                            supabase.table("raw_urls").insert({
                                'url': row['url'],
                                'target': row['target']
                            }).execute()
                        except Exception as single_error:
                            print(f"[ERROR] Échec sur URL: {row['url']} - {single_error}")

            print(f"[SUCCESS] {len(new_data)} URLs ajoutées depuis {file}")

        except Exception as e:
            print(f"[ERROR] Échec du fichier {file}: {e}")

# 2️⃣ Charger les données depuis Supabase vers Pandas
def load_all_from_supabase():
    """Loads ALL unprocessed URLs from Supabase with efficient pagination"""
    print("[INFO] Loading ALL unprocessed URLs from Supabase...")
    
    all_data = []
    page_size = 1000  # Optimal batch size for Supabase
    total_loaded = 0
    retry_count = 0
    max_retries = 3
    
    try:
        # First get total count (more efficient than loading all data)
        count_response = supabase.table("raw_urls") \
            .select("count", count="exact") \
            .eq("processed", False) \
            .execute()
        
        total_records = count_response.count
        if not total_records:
            print("[INFO] No unprocessed URLs found.")
            return pd.DataFrame()
        
        print(f"[INFO] Found {total_records} unprocessed URLs. Loading in batches...")

        # Progress tracking
        progress_interval = max(10, total_records // 20)  # Print progress every 5% or every 10 batches
        progress_step = max(1, progress_interval // page_size)  # Assurer qu'on ne divise pas par zéro

        # Paginated loading
        for offset in range(0, total_records, page_size):
            while retry_count < max_retries:
                try:
                    response = supabase.table("raw_urls") \
                        .select("*") \
                        .eq("processed", False) \
                        .range(offset, offset + page_size - 1) \
                        .execute()
                    
                    all_data.extend(response.data)
                    total_loaded += len(response.data)
                    retry_count = 0  # Reset retry counter after success
                    
                    # Progress reporting
                    if (offset // page_size) % progress_step == 0:
                        print(f"[PROGRESS] Loaded {total_loaded}/{total_records} ({total_loaded/total_records:.1%})")
                    
                    break  # Exit retry loop on success
                
                except Exception as e:
                    retry_count += 1
                    print(f"[WARNING] Batch {offset}-{offset+page_size} failed (attempt {retry_count}/{max_retries}): {str(e)}")
                    if retry_count >= max_retries:
                        print(f"[ERROR] Failed to load batch {offset}-{offset+page_size} after {max_retries} attempts")
                        raise
                    time.sleep(2 ** retry_count)  # Exponential backoff
        
        df = pd.DataFrame(all_data)
        print(f"[SUCCESS] Loaded all {len(df)} URLs from Supabase.")
        return df
    
    except Exception as e:
        print(f"[CRITICAL] Failed to load data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure
#

# 3️⃣ Nettoyage des URLs
def clean_urls(df):
    """Nettoie les URLs en supprimant les valeurs vides et les espaces."""
    print("[INFO] Nettoyage des URLs...")
    df = df.dropna(subset=['url'])
    df['url'] = df['url'].str.strip()
    print(f"[SUCCESS] {df.shape[0]} URLs après nettoyage.")
    return df

# 5️⃣ Extraction des features
import re
from urllib.parse import urlparse

def extract_features(df):
    """Extrait les features lexicales listées dans le premier tableau pour la détection de phishing."""
    print("[INFO] Extraction des features...")
    df['url'] = df['url'].str.strip().str.lower()
    parsed_urls = df['url'].apply(urlparse)

    df['url_length'] = df['url'].apply(len)
    df['num_special_chars'] = df['url'].apply(lambda x: len(re.findall(r'[;_?=&]', x)))
    df['digit_to_letter_ratio'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x) / (sum(c.isalpha() for c in x) + 1))
    
    parsed_urls = df['url'].apply(urlparse)
    
    df['top_level_domain'] = parsed_urls.apply(lambda x: x.netloc.split('.')[-1])
    df['primary_domain'] = parsed_urls.apply(lambda x: x.hostname if x.hostname else '')

    # Convert boolean features to SMALLINT (0 or 1)
    df['contains_ip'] = df['primary_domain'].apply(lambda x: int(x.replace('.', '').isdigit()))
    df['primary_domain_length'] = df['primary_domain'].apply(len)
    df['num_digits_primary_domain'] = df['primary_domain'].apply(lambda x: sum(c.isdigit() for c in x))
    df['num_non_alphanumeric_primary'] = df['primary_domain'].apply(lambda x: sum(not c.isalnum() for c in x))
    df['num_hyphens_primary'] = df['primary_domain'].apply(lambda x: x.count('-'))
    df['num_ats_primary'] = df['primary_domain'].apply(lambda x: x.count('@'))
    
    df['num_dots_subdomain'] = parsed_urls.apply(lambda x: x.netloc.count('.'))
    df['num_subdomains'] = parsed_urls.apply(lambda x: len(x.netloc.split('.')) - 2)
    
    df['num_double_slash'] = parsed_urls.apply(lambda x: x.path.count('//'))
    df['num_subdirectories'] = parsed_urls.apply(lambda x: x.path.count('/'))
    df['contains_encoded_space'] = parsed_urls.apply(lambda x: int('%20' in x.path))
    df['uppercase_dirs'] = parsed_urls.apply(lambda x: int(any(c.isupper() for c in x.path)))
    df['single_char_dirs'] = parsed_urls.apply(lambda x: int(any(len(part) == 1 for part in x.path.split('/'))))
    df['num_special_chars_path'] = parsed_urls.apply(lambda x: len(re.findall(r'[@_&=]', x.path)))
    df['num_zeroes_path'] = parsed_urls.apply(lambda x: x.path.count('0'))
    df['uppercase_ratio'] = parsed_urls.apply(lambda x: sum(c.isupper() for c in x.path) / (len(x.path) + 1))
    
    df['params_length'] = parsed_urls.apply(lambda x: len(x.query))
    df['num_queries'] = parsed_urls.apply(lambda x: x.query.count('&') + (1 if x.query else 0))

    print("[SUCCESS] Features extraites.")
    return df




# 6️⃣ Stocker les données finales dans Supabase
def save_to_supabase(df):
    """Stocke les nouvelles entrées par lots pour éviter les timeouts"""
    print("[INFO] Vérification des doublons et préparation des données...")
    
    try:
        # # 🔹 Supprimer url_hash si présent
        # if 'url_hash' in df.columns:
        #   df = df.drop(columns=['url_hash'])
        
        # 🔹 Récupérer les URLs existantes
        existing_urls_response = supabase.table("url_features").select("url").execute()
        existing_urls = {row["url"] for row in existing_urls_response.data} if existing_urls_response.data else set()
        df['url'] = df['url'].str.strip().str.lower()
        
        # 🔹 Filtrer les nouvelles URLs
        new_data = df[~df["url"].isin(existing_urls)]
        
        if new_data.empty:
            print("[INFO] Aucune nouvelle URL à insérer.")
            return
            
        # 🔹 Préparer les données pour l'insertion
        data_to_insert = new_data.drop(columns=['processed'], errors='ignore')
        
        # 🔹 Paramètres du traitement par lots
        batch_size = 500  # Taille optimale pour Supabase
        total_rows = len(data_to_insert)
        inserted_count = 0
        
        print(f"[INFO] Début de l'insertion de {total_rows} nouvelles URLs par lots de {batch_size}...")
        
        # 🔹 Traitement par lots
        for i in range(0, total_rows, batch_size):
            batch = data_to_insert.iloc[i:i+batch_size]
            batch_records = batch.to_dict(orient='records')
            
            try:
                supabase.table("url_features").insert(batch_records).execute()
                inserted_count += len(batch)
                print(f"[PROGRESS] Lot {i//batch_size + 1} inséré ({min(i+batch_size, total_rows)}/{total_rows} lignes)")
            except Exception as batch_error:
                print(f"[WARNING] Erreur sur le lot {i//batch_size + 1}: {batch_error}")
                # Optionnel: Réessayer avec un lot plus petit
                try:
                    smaller_batch_size = batch_size // 2
                    for j in range(0, len(batch), smaller_batch_size):
                        small_batch = batch.iloc[j:j+smaller_batch_size]
                        supabase.table("url_features").insert(small_batch.to_dict(orient='records')).execute()
                        inserted_count += len(small_batch)
                except Exception as retry_error:
                    print(f"[ERROR] Échec du réessai: {retry_error}")
                    continue
        
        print(f"[SUCCESS] {inserted_count}/{total_rows} nouvelles URLs stockées avec succès!")
        
        if inserted_count < total_rows:
            print(f"[WARNING] {total_rows - inserted_count} URLs n'ont pas pu être insérées")
            
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement: {e}")


def mark_as_processed(df):
     """Marks processed URLs as 'processed' in Supabase"""
     if df.empty:
        return  # No data to update
     urls = df["url"].tolist()  # Get the list of processed URLs
     for url in urls:
        supabase.table("raw_urls").update({"processed": True}).eq("url", url).execute()
     print(f"[SUCCESS] Marked {len(urls)} URLs as processed.")



# 🚀 Exécution du pipeline
def main_pipeline(file_paths):
    print("[START] Début du pipeline...")
    
    load_csv_to_supabase(file_paths)
    df = load_all_from_supabase()
    df = clean_urls(df)
    df = extract_features(df)
    save_to_supabase(df)
    mark_as_processed(df)
    
    
    print("[END] Pipeline terminé avec succès !")
