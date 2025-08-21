"""
Base classes pour les services.
Fournit des patterns communs pour la gestion d'erreurs, la validation et le logging.
"""
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum


# Configuration du logging
logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Exception de base pour les erreurs de service"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GENERIC_ERROR"
        self.details = details or {}


class ValidationError(ServiceError):
    """Exception pour les erreurs de validation"""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})
        self.field = field
        self.value = value


class ExternalServiceError(ServiceError):
    """Exception pour les erreurs de services externes"""
    
    def __init__(self, message: str, service: str, status_code: int = None):
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", {"service": service, "status_code": status_code})
        self.service = service
        self.status_code = status_code


@dataclass
class ServiceResponse:
    """Response standardisée pour les services"""
    success: bool
    data: Any = None
    error: str = None
    error_code: str = None
    details: Dict[str, Any] = None
    
    @classmethod
    def success_response(cls, data: Any = None) -> 'ServiceResponse':
        """Crée une réponse de succès"""
        return cls(success=True, data=data)
    
    @classmethod
    def error_response(cls, error: str, error_code: str = None, details: Dict[str, Any] = None) -> 'ServiceResponse':
        """Crée une réponse d'erreur"""
        return cls(success=False, error=error, error_code=error_code, details=details)


class LogLevel(Enum):
    """Niveaux de logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


T = TypeVar('T')


class BaseService(ABC):
    """
    Classe de base pour tous les services.
    Fournit des fonctionnalités communes pour la gestion d'erreurs, la validation et le logging.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        self.logger.info(f"Service {service_name} initialisé")
    
    def log(self, level: LogLevel, message: str, extra: Dict[str, Any] = None):
        """
        Log un message avec des informations contextuelles.
        
        Args:
            level: Niveau de log
            message: Message à logger
            extra: Informations supplémentaires
        """
        log_data = {
            "service": self.service_name,
        }
        if extra:
            log_data.update(extra)
        
        getattr(self.logger, level.value)(message, extra=log_data)
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """
        Valide que tous les champs requis sont présents.
        
        Args:
            data: Données à valider
            required_fields: Liste des champs requis
            
        Raises:
            ValidationError: Si un champ requis est manquant
        """
        for field in required_fields:
            if field not in data or data[field] is None:
                raise ValidationError(f"Champ requis manquant: {field}", field=field)
    
    def validate_field_type(self, value: Any, expected_type: type, field_name: str) -> None:
        """
        Valide le type d'un champ.
        
        Args:
            value: Valeur à valider
            expected_type: Type attendu
            field_name: Nom du champ
            
        Raises:
            ValidationError: Si le type ne correspond pas
        """
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"Type incorrect pour {field_name}: attendu {expected_type.__name__}, reçu {type(value).__name__}",
                field=field_name,
                value=value
            )
    
    def safe_execute(self, operation: callable, error_message: str = None, **kwargs) -> ServiceResponse:
        """
        Exécute une opération de manière sécurisée avec gestion d'erreurs.
        
        Args:
            operation: Fonction à exécuter
            error_message: Message d'erreur personnalisé
            **kwargs: Arguments à passer à la fonction
            
        Returns:
            ServiceResponse avec le résultat ou l'erreur
        """
        try:
            self.log(LogLevel.DEBUG, f"Exécution de l'opération: {operation.__name__}")
            result = operation(**kwargs)
            self.log(LogLevel.DEBUG, f"Opération {operation.__name__} réussie")
            return ServiceResponse.success_response(result)
        
        except ValidationError as e:
            self.log(LogLevel.WARNING, f"Erreur de validation: {e.message}", {"field": e.field, "value": e.value})
            return ServiceResponse.error_response(e.message, e.error_code, e.details)
        
        except ExternalServiceError as e:
            self.log(LogLevel.ERROR, f"Erreur de service externe: {e.message}", {"service": e.service, "status_code": e.status_code})
            return ServiceResponse.error_response(e.message, e.error_code, e.details)
        
        except ServiceError as e:
            self.log(LogLevel.ERROR, f"Erreur de service: {e.message}")
            return ServiceResponse.error_response(e.message, e.error_code, e.details)
        
        except Exception as e:
            error_msg = error_message or f"Erreur inattendue dans {operation.__name__}"
            self.log(LogLevel.ERROR, f"{error_msg}: {str(e)}", {"traceback": traceback.format_exc()})
            return ServiceResponse.error_response(error_msg, "UNEXPECTED_ERROR", {"original_error": str(e)})
    
    def normalize_score(self, score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
        """
        Normalise un score dans une plage donnée.
        
        Args:
            score: Score à normaliser
            min_score: Score minimum
            max_score: Score maximum
            
        Returns:
            Score normalisé
        """
        if score < min_score:
            return min_score
        if score > max_score:
            return max_score
        return score
    
    def truncate_text(self, text: str, max_length: int, suffix: str = "...") -> str:
        """
        Tronque un texte à une longueur maximale.
        
        Args:
            text: Texte à tronquer
            max_length: Longueur maximale
            suffix: Suffixe à ajouter si tronqué
            
        Returns:
            Texte tronqué
        """
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @abstractmethod
    def health_check(self) -> ServiceResponse:
        """
        Vérifie l'état de santé du service.
        Doit être implémenté par chaque service.
        
        Returns:
            ServiceResponse indiquant l'état de santé
        """
        pass


class CacheableService(BaseService):
    """
    Service avec capacités de cache.
    Fournit des méthodes pour la mise en cache des résultats.
    """
    
    def __init__(self, service_name: str, cache_ttl: int = 3600):
        super().__init__(service_name)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Génère une clé de cache basée sur les arguments.
        
        Args:
            *args: Arguments positionnels
            **kwargs: Arguments nommés
            
        Returns:
            Clé de cache
        """
        import hashlib
        import json
        
        # Créer une représentation sérialisable des arguments
        cache_data = {
            "args": args,
            "kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict, tuple))}
        }
        
        # Générer un hash de cette représentation
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Vérifie si une entrée de cache est encore valide.
        
        Args:
            key: Clé de cache
            
        Returns:
            True si valide, False sinon
        """
        import time
        
        if key not in self._cache_timestamps:
            return False
        
        return (time.time() - self._cache_timestamps[key]) < self.cache_ttl
    
    def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Valeur du cache ou None si pas trouvé/expiré
        """
        if key in self._cache and self._is_cache_valid(key):
            self.log(LogLevel.DEBUG, f"Cache hit pour la clé: {key}")
            return self._cache[key]
        
        # Nettoyer l'entrée expirée
        if key in self._cache:
            del self._cache[key]
            del self._cache_timestamps[key]
            self.log(LogLevel.DEBUG, f"Cache expiré pour la clé: {key}")
        
        return None
    
    def set_cache(self, key: str, value: Any) -> None:
        """
        Met une valeur en cache.
        
        Args:
            key: Clé de cache
            value: Valeur à mettre en cache
        """
        import time
        
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
        self.log(LogLevel.DEBUG, f"Valeur mise en cache pour la clé: {key}")
    
    def clear_cache(self) -> None:
        """Vide le cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.log(LogLevel.INFO, "Cache vidé")
    
    def cached_execute(self, operation: callable, cache_key: str = None, **kwargs) -> Any:
        """
        Exécute une opération avec mise en cache automatique.
        
        Args:
            operation: Fonction à exécuter
            cache_key: Clé de cache personnalisée
            **kwargs: Arguments à passer à la fonction
            
        Returns:
            Résultat de l'opération (du cache ou frais)
        """
        if cache_key is None:
            cache_key = self._generate_cache_key(operation.__name__, **kwargs)
        
        # Vérifier le cache
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Exécuter l'opération et mettre en cache
        result = operation(**kwargs)
        self.set_cache(cache_key, result)
        
        return result