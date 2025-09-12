import spacy
from fastapi import FastAPI, HTTPException
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Importation des routes de l'application
from .routes import contexts, history, generate, settings, messages, discussions, documents, search

# Importation du service Qdrant
from .services.qdrant_service import qdrant_service

from .database import create_or_update_collections

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialisation des modèles NLP

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restreindre si besoin, par exemple ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(contexts.router)
app.include_router(history.router)
app.include_router(settings.router)
app.include_router(generate.router)
app.include_router(discussions.router)
app.include_router(messages.router)
app.include_router(documents.router)
app.include_router(search.router)

# Récupération de l'URL d'Ollama depuis les variables d'environnement (pour un autre service)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# Route pour obtenir la version de l'application (utile pour le CI/CD)
@app.get("/version")
async def get_version():
    """Retourne la version de l'application basée sur le commit ID ou une version par défaut."""
    commit_id = os.getenv("COMMIT_ID", "dev")
    return {
        "version": commit_id,
        "status": "healthy"
    }

# Route de santé pour le CI/CD
@app.get("/health")
async def health_check():
    """Point de terminaison de vérification de santé pour le CI/CD."""
    return {
        "status": "healthy",
        "service": "backend-multi-llm",
        "timestamp": "2024-08-20T09:49:00Z"
    }

# Route de test pour vérifier la connexion à Qdrant
@app.get("/")
async def read_root():
    try:
        # Vérification de la connexion à Qdrant en récupérant des informations sur le serveur
        collections_info = qdrant_service.client.get_collections()
        logger.info(f"Connecté à Qdrant")
        logger.info(f"Connecté à Qdrant : {collections_info}")

        # Créer ou mettre à jour les collections (uniquement "documents" en hybride)
        summary = create_or_update_collections([
            {"name": "contexts", "vector_size": 768},
            {"name": "discussions", "vector_size": 768},
            {"name": "history", "vector_size": 768},
            {"name": "settings", "vector_size": 1},
            {"name": "messages", "vector_size": 768},
            {"name": "documents", "vector_size": 768},
        ])
 
        return {
            "status": "success",
            "message": "Collections initialisées avec succès (documents en mode hybride)",
            "collections": summary
        }
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Qdrant : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur de connexion à Qdrant")
