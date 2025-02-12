from fastapi import FastAPI, HTTPException
import os
import logging

from .services.mongo_service import mongo_service
from .routes import contexts, history, generate, settings, discussions

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mets ["http://localhost:5173"] si tu veux restreindre au frontend uniquement
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)

# Inclusion des routes
app.include_router(contexts.router)
app.include_router(history.router)
app.include_router(settings.router)
app.include_router(generate.router)
app.include_router(discussions.router)


# Charger l'URL d'Ollama depuis les variables d'environnement
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")


# Route de test MongoDB
@app.get("/")
async def read_root():
    try:
        collections = mongo_service.db.list_collection_names()
        return {"status": "success", "collections": collections}
    except Exception as e:
        logger.error(f"Erreur MongoDB: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur de connexion à MongoDB")



