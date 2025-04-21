from supabase import create_client, Client
from typing import Dict, List, Optional

# Configuration (à mettre dans des variables d'environnement en production)
SUPABASE_URL = "https://zhjjyoktbzsjsrbmpguw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpoamp5b2t0YnpzanNyYm1wZ3V3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMzNjgwOTAsImV4cCI6MjA1ODk0NDA5MH0.KcARKd95VoGh2umRuJkl-r1mG25MFikFlbKl-9ojpeQ"  # Clé API depuis les paramètres Supabase

# Initialisation du client
def init_supabase() -> Client:
    """Initialise et retourne le client Supabase"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test de connexion
        supabase.table("URL_Rows").select("count", count="exact").execute()
        print("[SUCCÈS] Connecté à Supabase avec succès")
        return supabase
    except Exception as e:
        print(f"[ERREUR] Échec de la connexion à Supabase: {e}")
        raise
supabase=init_supabase()
# Opérations sur les tables


# # Exemple d'utilisation
# if __name__ == "__main__":
#     try:
#         # Initialisation
#         sb = init_supabase()
        
#         # Création des tables
        
        
#         # Exemple d'insertion
#         new_url = {
#             "url": "https://example.com",
#             "target": 0
#         }
#         insert_response = sb.table("raw_urls").insert(new_url).execute()
#         print(f"Insertion réussie: {insert_response.data}")
        
#         # Exemple de requête
#         urls = sb.table("raw_urls").select("*").execute()
#         print("URLs enregistrées:", urls.data)
        
#     except Exception as e:
#         print(f"Erreur lors de l'exécution: {e}")