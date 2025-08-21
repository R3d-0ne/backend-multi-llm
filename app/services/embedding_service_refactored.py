"""
Service d'embeddings refactorisé avec gestion d'erreurs améliorée et cache.
"""
import requests
from typing import List, Union, Optional, Dict, Any
from .base_service import CacheableService, LogLevel, ServiceResponse, ExternalServiceError
from .config_service import config_service
from .model_loader import minilm_model
from ..libs.functions.global_functions import convert_numpy_types


class EmbeddingServiceRefactored(CacheableService):
    """
    Service d'embeddings refactorisé avec cache automatique et gestion d'erreurs robuste.
    """
    
    def __init__(self):
        super().__init__("EmbeddingService", cache_ttl=7200)  # Cache de 2 heures
        self.model = minilm_model
        self.timeout = getattr(config_service.llm, 'timeout', 30)
        
        self.log(LogLevel.INFO, f"Service d'embeddings initialisé avec le modèle: {self.model.model_name}")
    
    def get_embedding(self, texts: Union[str, List[str]], use_cache: bool = True) -> Union[List[float], List[List[float]], None]:
        """
        Génère des embeddings à partir d'un texte ou d'une liste de textes.
        
        Args:
            texts: Texte ou liste de textes pour lesquels générer des embeddings
            use_cache: Utiliser le cache pour éviter de recalculer des embeddings identiques
            
        Returns:
            - Si un seul texte: Liste de floats (vecteur d'embedding)
            - Si plusieurs textes: Liste de listes de floats
            - None en cas d'erreur
        """
        # Validation des entrées
        if not texts:
            self.log(LogLevel.WARNING, "Texte vide fourni pour l'embedding")
            return None
        
        # Normaliser l'entrée en liste
        is_single_text = isinstance(texts, str)
        if is_single_text:
            texts = [texts]
        
        self.validate_field_type(texts, list, "texts")
        
        # Traitement avec cache si demandé
        if use_cache:
            return self.cached_execute(
                self._compute_embeddings,
                texts=texts,
                is_single_text=is_single_text
            )
        else:
            return self._compute_embeddings(texts=texts, is_single_text=is_single_text)
    
    def _compute_embeddings(self, texts: List[str], is_single_text: bool) -> Union[List[float], List[List[float]], None]:
        """
        Calcule les embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes
            is_single_text: True si l'entrée originale était un seul texte
            
        Returns:
            Embeddings calculés
        """
        try:
            embeddings = []
            
            for i, text in enumerate(texts):
                self.log(LogLevel.DEBUG, f"Calcul de l'embedding {i+1}/{len(texts)}")
                
                embedding = self._compute_single_embedding(text)
                
                if embedding is None:
                    self.log(LogLevel.WARNING, f"Embedding vide pour le texte {i+1}")
                    return None
                
                embeddings.append(embedding)
            
            self.log(LogLevel.INFO, f"Embeddings calculés avec succès pour {len(texts)} texte(s)")
            
            # Retourner le format approprié
            return embeddings[0] if is_single_text else embeddings
            
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors du calcul des embeddings: {str(e)}")
            return None
    
    def _compute_single_embedding(self, text: str) -> Optional[List[float]]:
        """
        Calcule l'embedding pour un seul texte.
        
        Args:
            text: Texte pour lequel calculer l'embedding
            
        Returns:
            Vecteur d'embedding ou None en cas d'erreur
            
        Raises:
            ExternalServiceError: Si le service d'embeddings n'est pas accessible
        """
        if not text.strip():
            self.log(LogLevel.WARNING, "Texte vide fourni")
            return None
        
        try:
            api_url = f"{self.model.base_url}/api/embeddings"
            payload = {
                "model": self.model.model_name,
                "prompt": text.strip()
            }
            
            self.log(LogLevel.DEBUG, f"Appel à l'API d'embeddings: {api_url}")
            
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if not response.ok:
                raise ExternalServiceError(
                    f"Erreur API d'embeddings: {response.status_code} - {response.text}",
                    "EmbeddingService",
                    response.status_code
                )
            
            # Extraction de l'embedding
            response_data = response.json()
            embedding = response_data.get("embedding", [])
            
            if not embedding:
                self.log(LogLevel.WARNING, "Embedding vide retourné par l'API")
                return None
            
            # Conversion et validation
            converted_embedding = convert_numpy_types(embedding)
            
            if not isinstance(converted_embedding, list) or not converted_embedding:
                self.log(LogLevel.WARNING, "Format d'embedding invalide")
                return None
            
            # Validation que tous les éléments sont des nombres
            try:
                float_embedding = [float(x) for x in converted_embedding]
                self.log(LogLevel.DEBUG, f"Embedding calculé: dimension {len(float_embedding)}")
                return float_embedding
            except (ValueError, TypeError):
                self.log(LogLevel.WARNING, "Embedding contient des valeurs non numériques")
                return None
                
        except requests.Timeout:
            raise ExternalServiceError(
                f"Timeout lors de l'appel au service d'embeddings (>{self.timeout}s)",
                "EmbeddingService"
            )
        except requests.RequestException as e:
            raise ExternalServiceError(
                f"Erreur de connexion au service d'embeddings: {str(e)}",
                "EmbeddingService"
            )
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur inattendue lors du calcul d'embedding: {str(e)}")
            raise
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10, use_cache: bool = True) -> Optional[List[List[float]]]:
        """
        Calcule les embeddings pour un grand nombre de textes par lots.
        
        Args:
            texts: Liste de textes
            batch_size: Taille des lots pour éviter la surcharge
            use_cache: Utiliser le cache
            
        Returns:
            Liste d'embeddings ou None en cas d'erreur
        """
        if not texts:
            return []
        
        self.log(LogLevel.INFO, f"Calcul d'embeddings par lots: {len(texts)} textes, taille de lot: {batch_size}")
        
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                self.log(LogLevel.DEBUG, f"Traitement du lot {i//batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
                
                batch_embeddings = self.get_embedding(batch, use_cache=use_cache)
                
                if batch_embeddings is None:
                    self.log(LogLevel.ERROR, f"Erreur lors du traitement du lot {i//batch_size + 1}")
                    return None
                
                # S'assurer que le résultat est une liste de listes
                if len(batch) == 1 and isinstance(batch_embeddings[0], (int, float)):
                    batch_embeddings = [batch_embeddings]
                
                all_embeddings.extend(batch_embeddings)
            
            self.log(LogLevel.INFO, f"Embeddings par lots terminés: {len(all_embeddings)} embeddings calculés")
            return all_embeddings
            
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors du calcul d'embeddings par lots: {str(e)}")
            return None
    
    def get_embedding_dimension(self) -> Optional[int]:
        """
        Obtient la dimension des embeddings du modèle.
        
        Returns:
            Dimension des embeddings ou None en cas d'erreur
        """
        try:
            # Test avec un texte simple
            test_embedding = self.get_embedding("test", use_cache=False)
            
            if test_embedding and isinstance(test_embedding, list):
                dimension = len(test_embedding)
                self.log(LogLevel.INFO, f"Dimension des embeddings: {dimension}")
                return dimension
            else:
                self.log(LogLevel.WARNING, "Impossible de déterminer la dimension des embeddings")
                return None
                
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors de la détermination de la dimension: {str(e)}")
            return None
    
    def health_check(self) -> ServiceResponse:
        """
        Vérifie l'état de santé du service d'embeddings.
        
        Returns:
            ServiceResponse avec l'état de santé
        """
        try:
            # Test simple d'embedding
            test_text = "test de santé du service"
            test_embedding = self.get_embedding(test_text, use_cache=False)
            
            if test_embedding and len(test_embedding) > 0:
                return ServiceResponse.success_response({
                    "status": "healthy",
                    "service": "EmbeddingService",
                    "model": self.model.model_name,
                    "embedding_dimension": len(test_embedding),
                    "api_url": f"{self.model.base_url}/api/embeddings"
                })
            else:
                return ServiceResponse.error_response(
                    "Service d'embeddings ne répond pas correctement",
                    "EMBEDDING_SERVICE_ERROR"
                )
                
        except ExternalServiceError as e:
            return ServiceResponse.error_response(
                f"Service d'embeddings inaccessible: {e.message}",
                "EMBEDDING_SERVICE_UNAVAILABLE",
                {"service": e.service, "status_code": e.status_code}
            )
        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur lors du test de santé: {str(e)}",
                "HEALTH_CHECK_ERROR"
            )
    
    def clear_cache(self) -> ServiceResponse:
        """
        Vide le cache des embeddings.
        
        Returns:
            ServiceResponse confirmant l'opération
        """
        try:
            super().clear_cache()
            return ServiceResponse.success_response({
                "message": "Cache des embeddings vidé",
                "service": "EmbeddingService"
            })
        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur lors du vidage du cache: {str(e)}",
                "CACHE_CLEAR_ERROR"
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le cache.
        
        Returns:
            Dictionnaire avec les statistiques du cache
        """
        return {
            "cache_size": len(self._cache),
            "cache_entries": list(self._cache.keys())[:10],  # Premières 10 clés pour debug
            "cache_ttl": self.cache_ttl
        }


# Instance du service d'embeddings refactorisé
embedding_service_refactored = EmbeddingServiceRefactored()