"""
Couche de compatibilité pour permettre une migration progressive vers les services refactorisés.
Ce module permet d'utiliser les nouveaux services tout en maintenant la compatibilité avec l'API existante.
"""
import logging
from typing import Dict, List, Any, Optional, Union

# Import des services originaux
from .search_service import search_service as original_search_service
from .embedding_service import embedding_service as original_embedding_service
from .document_service import document_service as original_document_service
from .generate_service import generate_service as original_generate_service

# Import des services refactorisés
from .search_service_refactored import search_service_refactored
from .embedding_service_refactored import embedding_service_refactored
from .document_service_refactored import document_service_refactored
from .generate_service_refactored import generate_service_refactored

logger = logging.getLogger(__name__)


class ServiceMigrationManager:
    """
    Gestionnaire pour la migration progressive des services.
    Permet de basculer entre les anciens et nouveaux services selon la configuration.
    """
    
    def __init__(self):
        # Flags pour contrôler l'utilisation des nouveaux services
        self.use_refactored_search = True  # Activer par défaut pour tester
        self.use_refactored_embedding = True
        self.use_refactored_document = True
        self.use_refactored_generate = True
        
        logger.info("Gestionnaire de migration des services initialisé")
    
    def get_search_service(self):
        """Retourne le service de recherche approprié"""
        if self.use_refactored_search:
            return search_service_refactored
        return original_search_service
    
    def get_embedding_service(self):
        """Retourne le service d'embeddings approprié"""
        if self.use_refactored_embedding:
            return embedding_service_refactored
        return original_embedding_service
    
    def get_document_service(self):
        """Retourne le service de documents approprié"""
        if self.use_refactored_document:
            return document_service_refactored
        return original_document_service
    
    def get_generate_service(self):
        """Retourne le service de génération approprié"""
        if self.use_refactored_generate:
            return generate_service_refactored
        return original_generate_service
    
    def migrate_service(self, service_name: str, use_refactored: bool = True):
        """
        Bascule un service vers sa version refactorisée ou originale.
        
        Args:
            service_name: Nom du service ('search', 'embedding', 'document', 'generate')
            use_refactored: True pour utiliser la version refactorisée
        """
        if service_name == 'search':
            self.use_refactored_search = use_refactored
        elif service_name == 'embedding':
            self.use_refactored_embedding = use_refactored
        elif service_name == 'document':
            self.use_refactored_document = use_refactored
        elif service_name == 'generate':
            self.use_refactored_generate = use_refactored
        else:
            raise ValueError(f"Service inconnu: {service_name}")
        
        logger.info(f"Service {service_name} basculé vers {'refactorisé' if use_refactored else 'original'}")
    
    def get_migration_status(self) -> Dict[str, bool]:
        """Retourne l'état de migration de tous les services"""
        return {
            'search': self.use_refactored_search,
            'embedding': self.use_refactored_embedding,
            'document': self.use_refactored_document,
            'generate': self.use_refactored_generate
        }


# Instance globale du gestionnaire de migration
migration_manager = ServiceMigrationManager()


class CompatibilitySearchService:
    """
    Wrapper de compatibilité pour le service de recherche.
    Maintient l'API originale tout en utilisant le service refactorisé en arrière-plan.
    """
    
    def __init__(self):
        self.migration_manager = migration_manager
    
    def hybrid_search(self, *args, **kwargs):
        """Wrapper pour hybrid_search avec gestion des réponses"""
        service = self.migration_manager.get_search_service()
        
        if self.migration_manager.use_refactored_search:
            # Le service refactorisé retourne un ServiceResponse dans certains cas
            result = service.hybrid_search(*args, **kwargs)
            # S'assurer que le format de retour est compatible
            if hasattr(result, 'success') and hasattr(result, 'data'):
                return result.data if result.success else {"error": result.error, "results": []}
            return result
        else:
            return service.hybrid_search(*args, **kwargs)
    
    def search_with_generate_service(self, *args, **kwargs):
        """Wrapper pour search_with_generate_service"""
        service = self.migration_manager.get_search_service()
        
        if self.migration_manager.use_refactored_search:
            result = service.search_with_generate_service(*args, **kwargs)
            if hasattr(result, 'success') and hasattr(result, 'data'):
                return result.data if result.success else {"error": result.error}
            return result
        else:
            return service.search_with_generate_service(*args, **kwargs)


class CompatibilityEmbeddingService:
    """
    Wrapper de compatibilité pour le service d'embeddings.
    """
    
    def __init__(self):
        self.migration_manager = migration_manager
    
    def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]], None]:
        """Wrapper pour get_embedding"""
        service = self.migration_manager.get_embedding_service()
        
        if self.migration_manager.use_refactored_embedding:
            # Le service refactorisé a une signature compatible
            return service.get_embedding(texts)
        else:
            return service.get_embedding(texts)


class CompatibilityDocumentService:
    """
    Wrapper de compatibilité pour le service de documents.
    """
    
    def __init__(self):
        self.migration_manager = migration_manager
    
    def get_document(self, document_id: str):
        """Wrapper pour get_document"""
        service = self.migration_manager.get_document_service()
        
        if self.migration_manager.use_refactored_document:
            result = service.get_document(document_id)
            if hasattr(result, 'success') and hasattr(result, 'data'):
                if result.success:
                    return result.data
                else:
                    raise ValueError(result.error)
            return result
        else:
            return service.get_document(document_id)
    
    def list_documents(self):
        """Wrapper pour list_documents"""
        service = self.migration_manager.get_document_service()
        
        if self.migration_manager.use_refactored_document:
            result = service.list_documents()
            if hasattr(result, 'success') and hasattr(result, 'data'):
                return result.data.get("documents", []) if result.success else []
            return result
        else:
            return service.list_documents()
    
    def delete_document(self, document_id: str):
        """Wrapper pour delete_document"""
        service = self.migration_manager.get_document_service()
        
        if self.migration_manager.use_refactored_document:
            result = service.delete_document(document_id)
            if hasattr(result, 'success'):
                if result.success:
                    return True
                else:
                    raise ValueError(result.error)
            return result
        else:
            return service.delete_document(document_id)
    
    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]):
        """Wrapper pour update_document_metadata"""
        service = self.migration_manager.get_document_service()
        
        if self.migration_manager.use_refactored_document:
            result = service.update_document_metadata(document_id, metadata)
            if hasattr(result, 'success') and hasattr(result, 'data'):
                if result.success:
                    return result.data
                else:
                    raise ValueError(result.error)
            return result
        else:
            return service.update_document_metadata(document_id, metadata)


class CompatibilityGenerateService:
    """
    Wrapper de compatibilité pour le service de génération.
    """
    
    def __init__(self):
        self.migration_manager = migration_manager
    
    def generate_response(self, *args, **kwargs):
        """Wrapper pour generate_response"""
        service = self.migration_manager.get_generate_service()
        
        if self.migration_manager.use_refactored_generate:
            result = service.generate_response(*args, **kwargs)
            if hasattr(result, 'success') and hasattr(result, 'data'):
                if result.success:
                    return result.data
                else:
                    # Convertir l'erreur en exception pour maintenir la compatibilité
                    from fastapi import HTTPException
                    raise HTTPException(status_code=500, detail=result.error)
            return result
        else:
            return service.generate_response(*args, **kwargs)


# Instances de compatibilité qui peuvent remplacer les services originaux
search_service_compat = CompatibilitySearchService()
embedding_service_compat = CompatibilityEmbeddingService()
document_service_compat = CompatibilityDocumentService()
generate_service_compat = CompatibilityGenerateService()


def get_service_health_summary() -> Dict[str, Any]:
    """
    Retourne un résumé de l'état de santé de tous les services.
    """
    summary = {
        "migration_status": migration_manager.get_migration_status(),
        "health_checks": {}
    }
    
    # Tester la santé des services refactorisés
    refactored_services = {
        "search": search_service_refactored,
        "embedding": embedding_service_refactored,
        "document": document_service_refactored,
        "generate": generate_service_refactored
    }
    
    for name, service in refactored_services.items():
        try:
            if hasattr(service, 'health_check'):
                health = service.health_check()
                summary["health_checks"][f"{name}_refactored"] = {
                    "status": "healthy" if health.success else "unhealthy",
                    "details": health.data if health.success else health.error
                }
            else:
                summary["health_checks"][f"{name}_refactored"] = {
                    "status": "no_health_check"
                }
        except Exception as e:
            summary["health_checks"][f"{name}_refactored"] = {
                "status": "error",
                "error": str(e)
            }
    
    return summary


def run_migration_tests() -> Dict[str, Any]:
    """
    Exécute des tests basiques pour vérifier que la migration fonctionne.
    """
    test_results = {}
    
    # Test du service d'embeddings
    try:
        test_embedding = embedding_service_compat.get_embedding("test")
        test_results["embedding"] = {
            "success": test_embedding is not None,
            "details": f"Embedding dimension: {len(test_embedding) if test_embedding else 0}"
        }
    except Exception as e:
        test_results["embedding"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test du service de recherche
    try:
        test_search = search_service_compat.hybrid_search("test", limit=1)
        test_results["search"] = {
            "success": isinstance(test_search, dict),
            "details": f"Results returned: {len(test_search.get('results', []))}"
        }
    except Exception as e:
        test_results["search"] = {
            "success": False,
            "error": str(e)
        }
    
    # Test du service de documents
    try:
        test_docs = document_service_compat.list_documents()
        test_results["document"] = {
            "success": isinstance(test_docs, (list, dict)),
            "details": f"Documents found: {len(test_docs) if isinstance(test_docs, list) else 'dict returned'}"
        }
    except Exception as e:
        test_results["document"] = {
            "success": False,
            "error": str(e)
        }
    
    return test_results