import logging
import hashlib
import time
from typing import List, Union, Optional
from .model_loader import embedding_model
from ..libs.functions.global_functions import convert_numpy_types

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.model = embedding_model
        self.cache = {}  # Cache simple pour les embeddings
        self.cache_max_size = 1000  # Taille maximale du cache
        self.request_timeout = 30  # Timeout pour les requêtes
        self.max_retries = 3  # Nombre de tentatives en cas d'échec
        self.vector_size = 384  # all-MiniLM-L6-v2 output dimension

    def _get_cache_key(self, text: str) -> str:
        """Génère une clé de cache pour un texte."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _clean_cache(self):
        """Nettoie le cache s'il devient trop grand."""
        if len(self.cache) > self.cache_max_size:
            # Supprimer les 20% les plus anciens
            items_to_remove = int(self.cache_max_size * 0.2)
            oldest_keys = list(self.cache.keys())[:items_to_remove]
            for key in oldest_keys:
                del self.cache[key]
            logger.info(f"Cache nettoyé: {items_to_remove} entrées supprimées")

    def get_embedding(self, texts: Union[str, List[str]]) -> Optional[Union[List[float], List[List[float]]]]:
        """
        Génère des embeddings à partir d'un texte ou d'une liste de textes.
        Version améliorée avec cache et gestion d'erreurs robuste.
        
        Args:
            texts: Texte unique ou liste de textes
            
        Returns:
            Vecteur d'embedding unique ou liste de vecteurs
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
 
        if not texts or not isinstance(texts, list):
            logger.error("Textes invalides fournis à get_embedding")
            return None

        embeddings = []
        texts_to_process = []
        cache_hits = 0

        # Vérifier le cache pour chaque texte
        for text in texts:
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Texte vide ou invalide ignoré")
                embeddings.append(None)
                continue
                
            cache_key = self._get_cache_key(text.strip())
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
                cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                texts_to_process.append((len(embeddings) - 1, text.strip(), cache_key))

        logger.info(f"Cache hits: {cache_hits}/{len(texts)}")

        # Traiter les textes non mis en cache
        for index, text, cache_key in texts_to_process:
            embedding = self._get_embedding_local_with_retry(text)
            if embedding is not None:
                embeddings[index] = embedding
                # Mettre en cache
                self.cache[cache_key] = embedding
                self._clean_cache()
            else:
                logger.error(f"Échec de génération d'embedding pour le texte: {text[:100]}...")
                # Utiliser un vecteur zéro comme fallback
                embeddings[index] = [0.0] * self.vector_size

        # Filtrer les embeddings None (ne devrait plus arriver)
        embeddings = [emb for emb in embeddings if emb is not None]
        
        if not embeddings:
            logger.error("Aucun embedding généré")
            return None

        return embeddings[0] if single_text else embeddings

    def _get_embedding_local_with_retry(self, text: str) -> Optional[List[float]]:
        """Génère un embedding local (Sentence-Transformers) avec retry."""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Tentative {attempt + 1}/{self.max_retries} pour l'embedding local")
                vec = self.model.encode(text, normalize_embeddings=True)
                if vec is not None:
                    return convert_numpy_types(vec.tolist())
            except Exception as e:
                logger.warning(f"Erreur d'embedding locale tentative {attempt + 1}: {e}")
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Attente de {wait_time}s avant la prochaine tentative...")
                time.sleep(wait_time)
        logger.error(f"Échec de génération d'embedding locale après {self.max_retries} tentatives")
        return None

    def get_cache_stats(self) -> dict:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self.cache),
            "cache_max_size": self.cache_max_size,
            "cache_usage_percent": (len(self.cache) / self.cache_max_size) * 100
        }

    def clear_cache(self):
        """Vide le cache."""
        self.cache.clear()
        logger.info("Cache vidé")

# Singleton
embedding_service = EmbeddingService()
