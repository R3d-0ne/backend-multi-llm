from fastapi import APIRouter, HTTPException, Body, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from ..services.search_service import search_service

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

class DocumentQueryRequest(BaseModel):
    document_name: str = Field(..., description="Nom du document à interroger", min_length=1)
    question: str = Field(..., description="Question à poser sur le document", min_length=2)
    discussion_id: Optional[str] = Field(None, description="ID de la discussion (optionnel)")
    settings_id: Optional[str] = Field(None, description="ID des paramètres à utiliser (optionnel)")
    include_context: Optional[bool] = Field(True, description="Inclure le contexte de la discussion")
    max_context_length: Optional[int] = Field(5000, description="Longueur maximale du contexte", ge=1000, le=20000)


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
    Utilise la recherche par nom de document pour une meilleure performance.
    
    Returns:
        Liste des documents correspondant à la requête
    """
    try:
        # Utiliser la recherche efficace par nom de document
        results = search_service.search_document_by_name_efficient(
            document_name=q,
            limit=limit
        )
        
        # Si pas de résultats avec la recherche par nom, essayer une recherche vectorielle simple
        if not results.get("results"):
            results = search_service.simple_vector_search(
                query=q,
                limit=limit
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


@router.post("/search/document-query/")
async def query_document(request: DocumentQueryRequest = Body(...)):
    """
    Interroge un document spécifique par son nom et génère une réponse contextuelle.
    
    Cette API permet de :
    - Rechercher un document par son nom (recherche exacte ou partielle)
    - Poser une question spécifique sur ce document
    - Obtenir une réponse contextuelle basée sur le contenu du document
    - Intégrer le contexte de la discussion en cours si fourni
    
    Returns:
        Réponse générée par le LLM basée sur le contenu du document trouvé
    """
    try:
        # Rechercher le document par nom (recherche simplifiée)
        document_search_results = search_service.search_document_by_name_simple(
            document_name=request.document_name,
            limit=3  # Récupérer quelques résultats pour avoir des options
        )
        
        if not document_search_results or not document_search_results.get("results"):
            return {
                "error": f"Aucun document trouvé avec le nom '{request.document_name}'",
                "response": f"Je n'ai pas trouvé de document correspondant à '{request.document_name}'. Vérifiez le nom du document ou essayez une recherche partielle.",
                "document_found": False,
                "suggestions": []
            }
        
        # Récupérer le document trouvé
        found_document = document_search_results["results"][0]
        
        # Générer la réponse contextuelle
        response = search_service.generate_document_response(
            document=found_document,
            question=request.question,
            discussion_id=request.discussion_id,
            settings_id=request.settings_id,
            include_context=request.include_context,
            max_context_length=request.max_context_length
        )
        
        return {
            "response": response["answer"],
            "document_found": True,
            "document_info": {
                "id": found_document.get("id", ""),
                "filename": found_document.get("payload", {}).get("filename", ""),
                "score": found_document.get("score", 0),
                "metadata": found_document.get("payload", {})
            },
            "context_used": response.get("context_used", False),
            "discussion_context": response.get("discussion_context", False)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'interrogation du document: {str(e)}")


@router.get("/search/test-document-search/{document_name}")
async def test_document_search(document_name: str):
    """
    Endpoint de test pour vérifier la recherche de documents par nom.
    """
    try:
        # Test de la recherche efficace
        results = search_service.search_document_by_name_efficient(
            document_name=document_name,
            limit=5
        )
        
        return {
            "document_name": document_name,
            "results": results,
            "test_status": "success"
        }
    except Exception as e:
        return {
            "document_name": document_name,
            "error": str(e),
            "test_status": "failed"
        }

@router.get("/search/test-simple-search/{query}")
async def test_simple_search(query: str):
    """
    Endpoint de test pour vérifier la recherche vectorielle simple.
    """
    try:
        # Test de la recherche vectorielle simple
        results = search_service.simple_vector_search(
            query=query,
            limit=5
        )
        
        return {
            "query": query,
            "results": results,
            "test_status": "success"
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "test_status": "failed"
        }

@router.get("/search/debug-collection")
async def debug_collection():
    """
    Endpoint de debug pour voir le contenu de la collection.
    """
    try:
        # Récupérer quelques documents de la collection
        response = search_service.qdrant_service.client.scroll(
            collection_name="documents",
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        points = response[0]
        documents_info = []
        
        for point in points:
            payload = point.payload if point.payload else {}
            documents_info.append({
                "id": point.id,
                "filename": payload.get("filename", "N/A"),
                "has_cleaned_text": "cleaned_text" in payload,
                "payload_keys": list(payload.keys()) if payload else []
            })
        
        return {
            "collection_name": "documents",
            "total_documents": len(points),
            "sample_documents": documents_info,
            "test_status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "test_status": "failed"
        }
