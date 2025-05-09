from supabase import create_client, Client
from typing import Dict, List, Optional

# Configuration (à mettre dans des variables d'environnement en production)
SUPABASE_URL = "your_Supa_url"
SUPABASE_KEY = "you_key"  # Clé API depuis les paramètres Supabase

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



