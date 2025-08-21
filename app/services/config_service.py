"""
Configuration management service for the multi-LLM backend.
Centralizes all configuration management and environment variable handling.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()


@dataclass
class LLMConfig:
    """Configuration pour le LLM"""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    reranker_system_prompt: str
    general_system_prompt: str


@dataclass
class SearchConfig:
    """Configuration pour le service de recherche"""
    default_collection: str
    default_limit: int
    max_document_length: int
    keyword_boost_max: float
    llm_score_weight: float
    original_score_weight: float
    proximity_adjustment_max: float
    initial_limit_multiplier: int


@dataclass
class QdrantConfig:
    """Configuration pour Qdrant"""
    host: str
    port: int
    timeout: int


class ConfigService:
    """
    Service centralisé de gestion de la configuration.
    Fournit un accès uniforme à toutes les configurations de l'application.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implémentation du pattern Singleton"""
        if cls._instance is None:
            cls._instance = super(ConfigService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load_configuration()
            self._initialized = True
    
    def _load_configuration(self):
        """Charge toute la configuration depuis les variables d'environnement"""
        # Configuration LLM
        self.llm = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://host.docker.internal:11434"),
            model=os.getenv("LLM_MODEL", "llama3.1:8b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            reranker_system_prompt=self._get_reranker_prompt(),
            general_system_prompt=self._get_general_prompt()
        )
        
        # Configuration de recherche
        self.search = SearchConfig(
            default_collection=os.getenv("DEFAULT_COLLECTION", "documents"),
            default_limit=int(os.getenv("DEFAULT_SEARCH_LIMIT", "10")),
            max_document_length=int(os.getenv("MAX_DOCUMENT_LENGTH", "2000")),
            keyword_boost_max=float(os.getenv("KEYWORD_BOOST_MAX", "0.4")),
            llm_score_weight=float(os.getenv("LLM_SCORE_WEIGHT", "0.6")),
            original_score_weight=float(os.getenv("ORIGINAL_SCORE_WEIGHT", "0.4")),
            proximity_adjustment_max=float(os.getenv("PROXIMITY_ADJUSTMENT_MAX", "0.2")),
            initial_limit_multiplier=int(os.getenv("INITIAL_LIMIT_MULTIPLIER", "4"))
        )
        
        # Configuration Qdrant
        self.qdrant = QdrantConfig(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )
        
        logger.info("Configuration chargée avec succès")
    
    def _get_reranker_prompt(self) -> str:
        """Retourne le prompt système pour le réordonnancement LLM"""
        return """Tu es un assistant spécialisé dans l'évaluation de la pertinence des documents. Ta tâche est d'attribuer un score de 0 à 10 à chaque document en fonction de sa pertinence par rapport à la requête.

Critères d'évaluation :
1. Correspondance sémantique (0-4 points)
   - Pertinence générale du contenu
   - Présence des concepts clés
   - Cohérence avec la requête

2. Précision des informations (0-3 points)
   - Exactitude des détails
   - Actualité des informations
   - Fiabilité des sources

3. Exhaustivité (0-3 points)
   - Couverture du sujet
   - Profondeur des informations
   - Complémentarité avec la requête

Instructions :
- Score 0-2 : Document non pertinent ou hors sujet
- Score 3-5 : Document vaguement pertinent
- Score 6-7 : Document pertinent avec quelques informations utiles
- Score 8-10 : Document très pertinent et complet

Retourne uniquement un nombre entier de 0 à 10 sans autre texte ou explication."""
    
    def _get_general_prompt(self) -> str:
        """Retourne le prompt système général pour la génération de réponse"""
        return """Tu es un assistant IA spécialisé dans l'analyse de documents. Ta tâche est de :

1. Analyser précisément le contenu des documents fournis
2. Répondre uniquement en te basant sur les informations présentes dans ces documents
3. Ne pas faire d'hypothèses ou de suppositions non vérifiées
4. Si une information n'est pas claire ou manquante, le dire explicitement
5. Citer les documents sources pour chaque information importante
6. Ne pas confondre les documents entre eux
7. Être particulièrement attentif aux détails et aux nuances
8. Si tu n'es pas sûr d'une information, le dire clairement
9. Éviter les généralisations non justifiées
10. Privilégier la précision à la quantité d'informations

Format de réponse attendu :
- Réponse directe et précise à la question
- Citations des documents sources pour chaque information
- Précision si une information est manquante ou ambiguë
- Distinction claire entre les faits et les interprétations."""
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retourne toute la configuration sous forme de dictionnaire"""
        return {
            "llm": {
                "base_url": self.llm.base_url,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout
            },
            "search": {
                "default_collection": self.search.default_collection,
                "default_limit": self.search.default_limit,
                "max_document_length": self.search.max_document_length,
                "keyword_boost_max": self.search.keyword_boost_max,
                "llm_score_weight": self.search.llm_score_weight,
                "original_score_weight": self.search.original_score_weight,
                "proximity_adjustment_max": self.search.proximity_adjustment_max,
                "initial_limit_multiplier": self.search.initial_limit_multiplier
            },
            "qdrant": {
                "host": self.qdrant.host,
                "port": self.qdrant.port,
                "timeout": self.qdrant.timeout
            }
        }
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """
        Met à jour une valeur de configuration.
        
        Args:
            section: Section de configuration (llm, search, qdrant)
            key: Clé de configuration
            value: Nouvelle valeur
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            if hasattr(self, section):
                config_obj = getattr(self, section)
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    logger.info(f"Configuration mise à jour: {section}.{key} = {value}")
                    return True
                else:
                    logger.warning(f"Clé de configuration non trouvée: {section}.{key}")
                    return False
            else:
                logger.warning(f"Section de configuration non trouvée: {section}")
                return False
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la configuration: {e}")
            return False


# Instance singleton du service de configuration
config_service = ConfigService()