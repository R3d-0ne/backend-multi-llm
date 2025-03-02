import spacy
from fastapi import FastAPI, HTTPException
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

# Importation des routes de l'application
from .routes import contexts, history, generate, settings, messages, discussions, documents
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


# Récupération de l'URL d'Ollama depuis les variables d'environnement (pour un autre service)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

# Route de test pour vérifier la connexion à Qdrant
@app.get("/")
async def read_root():
    try:
        # Vérification de la connexion à Qdrant en récupérant des informations sur le serveur
        collections_info = qdrant_service.client.get_collections()
        logger.info(f"Connecté à Qdrant")
        logger.info(f"Connecté à Qdrant : {collections_info}")

        # Créer ou mettre à jour les collections et récupérer un résumé
        summary = create_or_update_collections([
            {"name": "contexts", "vector_size": 384},
            {"name": "discussions", "vector_size": 384},
            {"name": "history", "vector_size": 384},
            {"name": "settings", "vector_size": 1},
            {"name": "messages", "vector_size": 384},
            {"name": "documents", "vector_size": 384},
        ])

        return {
            "status": "success",
            "message": f"Collections existantes:",
            "collections": summary
        }
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Qdrant : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur de connexion à Qdrant")
