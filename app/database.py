import os
import logging
import time
from pymongo import MongoClient
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "deepseek")

if not MONGO_URI:
    raise ValueError("❌ ERREUR: MONGO_URI n'est pas défini dans le .env ou mal chargé.")

# Attendre que MongoDB soit disponible
MAX_RETRIES = 10
for i in range(MAX_RETRIES):
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Force la connexion pour vérifier si MongoDB est prêt
        db = client[DB_NAME]
        logger.info(f"✅ Connexion à MongoDB réussie : {MONGO_URI}, base {DB_NAME}")
        break
    except Exception as e:
        logger.warning(f"⏳ Tentative {i + 1}/{MAX_RETRIES} : MongoDB pas encore prêt...")
        time.sleep(5)
else:
    raise Exception("❌ MongoDB n'est pas accessible après plusieurs tentatives.")


# Vérification et initialisation des collections
def init_db():
    """ Vérifie que les collections existent, sinon insère un doc temporaire """
    collections = ["history", "contexts", "settings", "models", "documents", "summaries", "discussions"]
    db_collections = db.list_collection_names()

    for collection in collections:
        if collection not in db_collections:
            db[collection].insert_one({"_init": True})  # Ajout d'un doc temporaire
            logger.info(f"✅ Collection '{collection}' initialisée.")


init_db()  # Exécute l'initialisation une seule fois au démarrage


# Fonction pour récupérer la base de données
def get_database():
    return db
