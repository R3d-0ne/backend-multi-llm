import logging
import requests
from .model_loader import minilm_model
from ..libs.functions.global_functions import convert_numpy_types

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = minilm_model

    def get_embedding(self, texts):
        """
        Génère des embeddings à partir d'un texte ou d'une liste de textes.
        - Si un seul texte est fourni (`str`), il est converti en liste.
        - Retourne une liste de vecteurs (un par texte).
        """
        if isinstance(texts, str):
            texts = [texts]
 
        if not texts or not isinstance(texts, list):
            return None

        try:
            embeddings = []
            for text in texts:
                logger.info(f"Envoi de la requête à {self.model.base_url}/api/embeddings")
                response = requests.post(
                    f"{self.model.base_url}/api/embeddings",
                    json={
                        "model": self.model.model_name,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                logger.info(f"Réponse reçue : {response.status_code}")
                
                embedding = response.json().get("embedding", [])
                if embedding:
                    logger.info(f"Dimension du vecteur d'embedding : {len(embedding)}")
                else:
                    logger.error("L'embedding retourné est vide")
                embeddings.append(convert_numpy_types(embedding))
            return embeddings[0] if len(texts) == 1 else embeddings
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des embeddings : {str(e)}")
            logger.error(f"URL appelée : {self.model.base_url}/api/embeddings")
            return None

# Singleton
embedding_service = EmbeddingService()
