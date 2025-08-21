from fastapi import APIRouter, HTTPException, Body, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from ..services.service_compatibility import migration_manager

router = APIRouter()

class SearchFilters(BaseModel):
    has_tables: Optional[bool] = Field(None, description="Filtre les documents contenant des tables")
    has_named_entities: Optional[bool] = Field(None, description="Filtre les documents contenant des entités nommées")
    has_dates: Optional[bool] = Field(None, description="Filtre les documents contenant des dates")
    has_money_amounts: Optional[bool] = Field(None, description="Filtre les documents contenant des montants d'argent")
    has_emails: Optional[bool] = Field(None, description="Filtre les documents contenant des emails")
    has_phone_numbers: Optional[bool] = Field(None, description="Filtre les documents contenant des numéros de téléphone")
    upload_date_range: Optional[Dict[str, str]] = Field(None, description="Filtre par plage de dates (format ISO)")
    filename_contains: Optional[str] = Field(None, description="Filtre les documents dont le nom contient la chaîne spécifiée")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Requête de recherche textuelle", min_length=2)
    limit: Optional[int] = Field(5, description="Nombre maximum de résultats", ge=1, le=20)
    filters: Optional[SearchFilters] = Field(None, description="Filtres à appliquer aux résultats")
    use_llm_reranking: Optional[bool] = Field(True, description="Active le réordonnancement des résultats par LLM")
    boost_keywords: Optional[bool] = Field(True, description="Augmente le score des résultats contenant des mots-clés de la requête")
    generate_answer: Optional[bool] = Field(False, description="Génère une réponse basée sur le contexte des documents trouvés")

class InternalSearchRequest(BaseModel):
    query: str = Field(..., description="Requête de recherche textuelle", min_length=2)
    discussion_id: Optional[str] = Field(None, description="ID de la discussion (optionnel)")
    settings_id: Optional[str] = Field(None, description="ID des paramètres à utiliser (optionnel)")
    limit: Optional[int] = Field(5, description="Nombre maximum de résultats", ge=1, le=10)
    filters: Optional[SearchFilters] = Field(None, description="Filtres à appliquer aux résultats")


def get_search_service():
    """Retourne le service de recherche approprié (refactorisé ou original)."""
    if COMPATIBILITY_AVAILABLE and hasattr(migration_manager, 'get_search_service'):
        return migration_manager.get_search_service()
    return search_service


@router.post("/search/")
async def search_documents(search_params: SearchRequest = Body(...)):
    """
    Recherche avancée de documents avec support LLM.
    
    Cette API combine:
    - Recherche sémantique par embeddings
    - Boost de mots-clés
    - Filtrage par métadonnées
    - Réordonnancement par LLM
    - Génération de réponses
    
    Returns:
        Liste des documents correspondant à la requête, avec scores et métadonnées
    """
    try:
        # Conversion des filtres de Pydantic à dict
        filters = {}
        if search_params.filters:
            filter_dict = search_params.filters.dict(exclude_none=True)
            for key, value in filter_dict.items():
                if key == "filename_contains" and value:
                    # Traitement spécial pour la recherche par nom de fichier
                    continue  # Ce sera traité séparément
                else:
                    filters[key] = value
        
        search_service = migration_manager.get_search_service()
        results = search_service.hybrid_search(
            query=search_params.query,
            limit=search_params.limit,
            filters=filters if filters else None,
            use_llm_reranking=search_params.use_llm_reranking,
            boost_keywords=search_params.boost_keywords,
            generate_answer=search_params.generate_answer
        )
        
        # Filtrage post-recherche par nom de fichier si nécessaire
        if search_params.filters and search_params.filters.filename_contains:
            filter_text = search_params.filters.filename_contains.lower()
            filtered_results = []
            for result in results.get("results", []):
                filename = result.get("payload", {}).get("filename", "").lower()
                if filter_text in filename:
                    filtered_results.append(result)
            results["results"] = filtered_results
            results["total_found"] = len(filtered_results)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")


@router.get("/search/simple")
async def simple_search(
    q: str = Query(..., description="Requête de recherche", min_length=2),
    limit: int = Query(5, description="Nombre maximum de résultats", ge=1, le=20),
    generate_answer: bool = Query(False, description="Génère une réponse basée sur le contexte des documents trouvés")
):
    """
    Version simplifiée de la recherche pour les requêtes GET simples.
    
    Returns:
        Liste des documents correspondant à la requête
    """
    try:
        search_service = migration_manager.get_search_service()
        results = search_service.hybrid_search(
            query=q,
            limit=limit,
            use_llm_reranking=True,
            boost_keywords=True,
            generate_answer=generate_answer
        )
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")


@router.post("/search/internal/")
async def internal_search(search_params: InternalSearchRequest = Body(...)):
    """
    Recherche interne dans les documents avec génération de réponse via le service de génération existant.
    
    Cette API combine:
    - La recherche dans les documents internes
    - L'utilisation du modèle de génération pour produire une réponse contextualisée
    - L'intégration avec le système de discussions existant
    
    Returns:
        La réponse générée par le LLM et les documents utilisés comme contexte
    """
    try:
        # Conversion des filtres de Pydantic à dict
        filters = {}
        if search_params.filters:
            filter_dict = search_params.filters.dict(exclude_none=True)
            for key, value in filter_dict.items():
                if key == "filename_contains" and value:
                    # Traitement spécial pour la recherche par nom de fichier
                    continue  # Ce sera traité séparément
                else:
                    filters[key] = value
        
        # Exécution de la recherche avec le service de génération
        search_service = migration_manager.get_search_service()
        results = search_service.search_with_generate_service(
            query=search_params.query,
            discussion_id=search_params.discussion_id,
            settings_id=search_params.settings_id,
            limit=search_params.limit,
            filters=filters if filters else None
        )
        
        # Filtrage post-recherche par nom de fichier si nécessaire
        if search_params.filters and search_params.filters.filename_contains:
            filter_text = search_params.filters.filename_contains.lower()
            filtered_results = []
            for result in results.get("search_results", []):
                filename = result.get("payload", {}).get("filename", "").lower()
                if filter_text in filename:
                    filtered_results.append(result)
            results["search_results"] = filtered_results
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche interne: {str(e)}") 