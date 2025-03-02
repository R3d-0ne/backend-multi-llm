import logging
from sentence_transformers import SentenceTransformer

from .model_loader import minilm_model

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        """
        Initialise le modèle d'embeddings depuis Hugging Face.
        - Utilise par défaut `all-MiniLM-L6-v2` (léger et rapide).
        """
        try:
            self.model = minilm_model
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle d'embeddings : {str(e)}")
            raise RuntimeError("Impossible de charger le modèle d'embeddings.")

    def get_embedding(self, texts):
        """
        Génère des embeddings à partir d'un texte ou d'une liste de textes.
        - Si un seul texte est fourni (`str`), il est converti en liste.
        - Retourne une liste de vecteurs (un par texte).
        """
        if isinstance(texts, str):
            texts = [texts]  # ✅ Convertir un texte unique en liste

        if not texts or not isinstance(texts, list):
            logger.error("❌ Entrée invalide pour l'embedding (doit être une liste de textes).")
            return None

        try:
            embeddings = self.model.encode(texts)  # ✅ Retourne un tableau numpy
            return embeddings  # ✅ Convertir en liste native Python
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des embeddings : {str(e)}")
            return None

# Singleton
embedding_service = EmbeddingService()
