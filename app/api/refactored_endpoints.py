"""
Exemple d'endpoints API utilisant les services refactorisés.
Démontre l'intégration des nouveaux services dans une API FastAPI.
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# Import des services refactorisés via la couche de compatibilité
from app.services.service_compatibility import (
    search_service_compat,
    embedding_service_compat,
    document_service_compat,
    generate_service_compat,
    get_service_health_summary,
    migration_manager
)

# Import direct des services refactorisés pour les fonctionnalités avancées
from app.services.search_service_refactored import search_service_refactored
from app.services.embedding_service_refactored import embedding_service_refactored
from app.services.document_service_refactored import document_service_refactored
from app.services.generate_service_refactored import generate_service_refactored


# Modèles de données
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    use_llm_reranking: bool = True
    boost_keywords: bool = True
    generate_answer: bool = False
    collection_name: Optional[str] = None


class EmbeddingRequest(BaseModel):
    texts: List[str]
    use_cache: bool = True
    batch_size: int = 10


class DocumentSearchRequest(BaseModel):
    search_criteria: Dict[str, Any]
    limit: int = 100


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None


# Création du router
router = APIRouter(prefix="/api/v2", tags=["Services Refactorisés"])


@router.get("/health")
async def get_system_health():
    """
    Endpoint pour vérifier l'état de santé de tous les services.
    """
    try:
        health_summary = get_service_health_summary()
        return {
            "status": "success",
            "data": health_summary,
            "message": "État de santé des services récupéré avec succès"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la vérification de santé: {str(e)}")


@router.post("/search/hybrid")
async def hybrid_search(request: SearchRequest):
    """
    Endpoint de recherche hybride utilisant le service refactorisé.
    """
    try:
        result = search_service_refactored.hybrid_search(
            query=request.query,
            limit=request.limit,
            filters=request.filters,
            use_llm_reranking=request.use_llm_reranking,
            boost_keywords=request.boost_keywords,
            generate_answer=request.generate_answer,
            collection_name=request.collection_name
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Recherche effectuée pour: {request.query}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")


@router.post("/search/with-generation")
async def search_with_generation(
    query: str,
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None
):
    """
    Endpoint de recherche avec génération de réponse.
    """
    try:
        result = search_service_refactored.search_with_generate_service(
            query=query,
            limit=limit,
            filters=filters
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Recherche avec génération effectuée pour: {query}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche avec génération: {str(e)}")


@router.post("/embeddings/single")
async def get_single_embedding(text: str, use_cache: bool = True):
    """
    Endpoint pour obtenir l'embedding d'un seul texte.
    """
    try:
        embedding = embedding_service_refactored.get_embedding(text, use_cache=use_cache)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="Impossible de générer l'embedding")
        
        return {
            "status": "success",
            "data": {
                "text": text,
                "embedding": embedding,
                "dimension": len(embedding)
            },
            "message": "Embedding généré avec succès"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération d'embedding: {str(e)}")


@router.post("/embeddings/batch")
async def get_batch_embeddings(request: EmbeddingRequest):
    """
    Endpoint pour obtenir les embeddings de plusieurs textes en lot.
    """
    try:
        embeddings = embedding_service_refactored.get_embeddings_batch(
            texts=request.texts,
            batch_size=request.batch_size,
            use_cache=request.use_cache
        )
        
        if embeddings is None:
            raise HTTPException(status_code=400, detail="Impossible de générer les embeddings")
        
        return {
            "status": "success",
            "data": {
                "texts": request.texts,
                "embeddings": embeddings,
                "count": len(embeddings),
                "dimension": len(embeddings[0]) if embeddings else 0
            },
            "message": f"Embeddings générés pour {len(request.texts)} textes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération d'embeddings: {str(e)}")


@router.get("/embeddings/stats")
async def get_embedding_stats():
    """
    Endpoint pour obtenir les statistiques du service d'embeddings.
    """
    try:
        health = embedding_service_refactored.health_check()
        cache_stats = embedding_service_refactored.get_cache_stats()
        
        return {
            "status": "success",
            "data": {
                "health": health.data if health.success else health.error,
                "cache_stats": cache_stats,
                "service_healthy": health.success
            },
            "message": "Statistiques du service d'embeddings"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques: {str(e)}")


@router.get("/documents")
async def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    file_type: Optional[str] = None
):
    """
    Endpoint pour lister les documents avec pagination.
    """
    try:
        # Construire les critères de filtrage si nécessaire
        filters = {}
        if file_type:
            filters["file_type"] = file_type
        
        result = document_service_refactored.list_documents(
            limit=limit,
            offset=offset,
            filter_criteria=filters if filters else None
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "status": "success",
            "data": result.data,
            "message": f"Documents listés (limite: {limit}, offset: {offset})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du listage des documents: {str(e)}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Endpoint pour récupérer un document spécifique.
    """
    try:
        result = document_service_refactored.get_document(document_id)
        
        if not result.success:
            raise HTTPException(status_code=404, detail=result.error)
        
        return {
            "status": "success",
            "data": result.data,
            "message": f"Document {document_id} récupéré"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du document: {str(e)}")


@router.get("/documents/stats")
async def get_document_stats():
    """
    Endpoint pour obtenir les statistiques des documents.
    """
    try:
        result = document_service_refactored.get_document_statistics()
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "status": "success",
            "data": result.data,
            "message": "Statistiques des documents récupérées"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques: {str(e)}")


@router.post("/documents/search")
async def search_documents(request: DocumentSearchRequest):
    """
    Endpoint pour rechercher des documents selon des critères.
    """
    try:
        result = document_service_refactored.search_documents(
            search_criteria=request.search_criteria,
            limit=request.limit
        )
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "status": "success",
            "data": result.data,
            "message": f"Recherche de documents effectuée avec {len(request.search_criteria)} critères"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche de documents: {str(e)}")


@router.post("/generate/validate")
async def validate_generation_request(request: GenerateRequest):
    """
    Endpoint pour valider une requête de génération sans l'exécuter.
    """
    try:
        result = generate_service_refactored.validate_generate_request(request.dict())
        
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)
        
        return {
            "status": "success",
            "data": result.data,
            "message": "Requête de génération validée"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la validation: {str(e)}")


@router.get("/generate/stats")
async def get_generation_stats():
    """
    Endpoint pour obtenir les statistiques du service de génération.
    """
    try:
        stats_result = generate_service_refactored.get_generation_statistics()
        health_result = generate_service_refactored.health_check()
        
        return {
            "status": "success",
            "data": {
                "statistics": stats_result.data if stats_result.success else stats_result.error,
                "health": health_result.data if health_result.success else health_result.error,
                "service_healthy": health_result.success
            },
            "message": "Statistiques du service de génération"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques: {str(e)}")


@router.post("/migration/toggle-service")
async def toggle_service_migration(
    service_name: str,
    use_refactored: bool = True
):
    """
    Endpoint pour basculer un service vers sa version refactorisée ou originale.
    """
    try:
        valid_services = ['search', 'embedding', 'document', 'generate']
        if service_name not in valid_services:
            raise HTTPException(status_code=400, detail=f"Service invalide. Services valides: {valid_services}")
        
        migration_manager.migrate_service(service_name, use_refactored)
        
        return {
            "status": "success",
            "data": {
                "service": service_name,
                "using_refactored": use_refactored,
                "migration_status": migration_manager.get_migration_status()
            },
            "message": f"Service {service_name} basculé vers {'refactorisé' if use_refactored else 'original'}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la migration: {str(e)}")


@router.get("/migration/status")
async def get_migration_status():
    """
    Endpoint pour obtenir l'état de migration de tous les services.
    """
    try:
        status = migration_manager.get_migration_status()
        
        return {
            "status": "success",
            "data": status,
            "message": "État de migration récupéré"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du statut: {str(e)}")


# Exemple d'utilisation dans une application FastAPI
"""
Pour utiliser ces endpoints dans votre application FastAPI:

from fastapi import FastAPI
from app.api.refactored_endpoints import router

app = FastAPI()
app.include_router(router)

# Les endpoints seront disponibles sous:
# - GET /api/v2/health
# - POST /api/v2/search/hybrid
# - POST /api/v2/embeddings/single
# - etc.
"""