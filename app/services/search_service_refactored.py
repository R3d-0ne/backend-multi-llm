"""
Service de recherche refactorisé utilisant une architecture modulaire.
Combine recherche vectorielle, filtrage par métadonnées et réordonnancement par LLM.
"""
import json
import requests
import traceback
from typing import List, Dict, Any, Optional

from .base_service import CacheableService, LogLevel, ServiceResponse, ExternalServiceError
from .config_service import config_service
from .search_components import (
    DocumentProcessor, KeywordBooster, LLMReranker, SearchResultFilter, SearchResult
)
from .qdrant_service import QdrantService
from .embedding_service import embedding_service
from ..libs.functions.global_functions import (
    tokenize_text, 
    remove_stopwords,
    create_sparse_vector,
    extract_entities_advanced
)


class SearchServiceRefactored(CacheableService):
    """
    Service de recherche avancée refactorisé avec architecture modulaire.
    Combine recherche vectorielle, filtrage et réordonnancement LLM.
    """
    
    def __init__(self):
        super().__init__("SearchService", cache_ttl=1800)  # Cache de 30 minutes
        
        # Configuration
        self.config = config_service.search
        self.llm_config = config_service.llm
        
        # Services
        self.qdrant_service = QdrantService()
        self.document_processor = DocumentProcessor()
        self.keyword_booster = KeywordBooster()
        self.llm_reranker = LLMReranker()
        self.result_filter = SearchResultFilter()
        
        self.log(LogLevel.INFO, f"Service de recherche initialisé avec la collection '{self.config.default_collection}'")
    
    def hybrid_search(
        self,
        query: str,
        limit: int = None,
        filters: Dict[str, Any] = None,
        use_llm_reranking: bool = True,
        boost_keywords: bool = True,
        generate_answer: bool = False,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Effectue une recherche hybride combinant recherche vectorielle et filtrage par métadonnées,
        avec réordonnancement optionnel par LLM.

        Args:
            query: Requête de recherche textuelle
            limit: Nombre maximum de résultats (par défaut config.default_limit)
            filters: Filtres à appliquer sur les métadonnées
            use_llm_reranking: Active le réordonnancement des résultats par LLM
            boost_keywords: Augmente le score des résultats contenant des mots-clés de la requête
            generate_answer: Génère une réponse basée sur le contexte des documents trouvés
            collection_name: Nom de la collection (par défaut config.default_collection)

        Returns:
            Dictionnaire contenant les résultats de recherche et optionnellement une réponse générée
        """
        # Validation des entrées
        self.validate_required_fields({"query": query}, ["query"])
        self.validate_field_type(query, str, "query")
        
        if limit is None:
            limit = self.config.default_limit
        
        if collection_name is None:
            collection_name = self.config.default_collection
        
        # Génération de la clé de cache
        cache_key = self._generate_cache_key(
            "hybrid_search",
            query=query,
            limit=limit,
            filters=filters,
            use_llm_reranking=use_llm_reranking,
            boost_keywords=boost_keywords,
            generate_answer=generate_answer,
            collection_name=collection_name
        )
        
        # Vérifier le cache
        cached_result = self.get_from_cache(cache_key)
        if cached_result is not None:
            self.log(LogLevel.DEBUG, "Résultat retourné depuis le cache")
            return cached_result
        
        # Exécuter la recherche
        def search_operation():
            return self._execute_hybrid_search(
                query, limit, filters, use_llm_reranking, 
                boost_keywords, generate_answer, collection_name
            )
        
        response = self.safe_execute(
            search_operation,
            f"Erreur lors de la recherche pour la requête: {query}"
        )
        
        if response.success:
            # Mettre en cache le résultat
            self.set_cache(cache_key, response.data)
            return response.data
        else:
            return {
                "error": response.error,
                "results": [],
                "total_found": 0,
                "query": query
            }
    
    def _execute_hybrid_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        use_llm_reranking: bool,
        boost_keywords: bool,
        generate_answer: bool,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Exécute la logique de recherche hybride.
        """
        self.log(LogLevel.INFO, f"Recherche hybride: '{query}' (limite: {limit}, collection: {collection_name})")
        
        # Étape 1: Recherche vectorielle initiale
        initial_limit = limit * self.config.initial_limit_multiplier if use_llm_reranking else limit
        search_results = self._perform_initial_search(query, initial_limit, filters, collection_name)
        
        if not search_results:
            return {"results": [], "total_found": 0, "query": query}
        
        # Étape 2: Filtrage (si pas déjà fait dans la recherche hybride)
        if filters:
            search_results = self.result_filter.apply_filters(search_results, filters)
        
        # Étape 3: Boost des mots-clés
        if boost_keywords:
            search_results = self.keyword_booster.boost_results(search_results, query)
        
        # Étape 4: Réordonnancement par LLM
        if use_llm_reranking:
            search_results = self.llm_reranker.rerank_results(search_results, query, limit)
        else:
            search_results = search_results[:limit]
        
        # Étape 5: Génération d'une réponse contextuelle si demandée
        response = {
            "results": search_results,
            "total_found": len(search_results),
            "query": query
        }
        
        if generate_answer:
            answer = self._generate_contextual_answer(search_results, query)
            response["generated_answer"] = answer
        
        return response
    
    def _perform_initial_search(
        self, 
        query: str, 
        limit: int,
        filters: Optional[Dict[str, Any]],
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Effectue la recherche vectorielle initiale.
        """
        # Pour la collection "documents", essayer d'abord la recherche hybride avancée
        if collection_name == "documents":
            try:
                self.log(LogLevel.DEBUG, "Tentative de recherche hybride avancée")
                results = self._hybrid_dense_sparse_search(query, limit, filters, collection_name)
                if results:
                    return results
            except Exception as e:
                self.log(LogLevel.WARNING, f"Échec de la recherche hybride avancée: {e}, fallback vers recherche simple")
        
        # Recherche vectorielle standard
        return self._standard_vector_search(query, limit, collection_name)
    
    def _hybrid_dense_sparse_search(
        self, 
        query: str, 
        limit: int,
        filters: Optional[Dict[str, Any]],
        collection_name: str
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride en combinant vecteurs denses, d'entités et épars.
        """
        self.log(LogLevel.DEBUG, f"Recherche hybride dense-sparse sur '{collection_name}'")
        
        # 1. Générer le vecteur dense pour la requête
        dense_vector = embedding_service.get_embedding(query)
        if dense_vector is None:
            raise ExternalServiceError("Impossible de générer l'embedding dense pour la requête", "EmbeddingService")
        
        # 2. Extraire les entités et générer leur embedding
        try:
            entities_result = extract_entities_advanced(query)
            entities = entities_result.get('entities', [])
            entity_text = ' '.join([ent['text'] for ent in entities])
            entity_vector = embedding_service.get_embedding(entity_text)
        except Exception as e:
            self.log(LogLevel.WARNING, f"Erreur lors de l'extraction d'entités: {e}")
            entity_vector = None
        
        if entity_vector is None:
            # Générer un vecteur vide de même taille que dense_vector
            if hasattr(dense_vector, "shape"):
                entity_vector = [0.0] * dense_vector.shape[0]
            else:
                entity_vector = [0.0] * len(dense_vector)
        
        # 3. Générer le vecteur épars
        tokens = tokenize_text(query)
        tokens_no_stopwords = remove_stopwords(tokens)
        sparse_vector = create_sparse_vector(tokens_no_stopwords, vocab_size=10000)
        
        # 4. Préparer les vecteurs de recherche avec pondération
        search_vectors = {
            "dense": {"vector": dense_vector, "weight": 0.6},
            "entity": {"vector": entity_vector, "weight": 0.3},
            "sparse": {"vector": sparse_vector, "weight": 0.1}
        }
        
        # 5. Effectuer la recherche hybride
        self.log(LogLevel.DEBUG, "Recherche hybride avec pondération: dense=0.6, entity=0.3, sparse=0.1")
        results = self.qdrant_service.hybrid_search(
            collection_name=collection_name,
            vectors=search_vectors,
            limit=limit,
            filter_conditions=filters
        )
        
        return results
    
    def _standard_vector_search(self, query: str, limit: int, collection_name: str) -> List[Dict[str, Any]]:
        """
        Effectue une recherche vectorielle standard.
        """
        self.log(LogLevel.DEBUG, f"Recherche vectorielle standard sur '{collection_name}'")
        
        # Obtenir le vecteur d'embedding pour la requête
        query_vector = embedding_service.get_embedding(query)
        if query_vector is None:
            raise ExternalServiceError("Impossible de générer un embedding pour la requête", "EmbeddingService")
        
        # Recherche dans Qdrant
        search_results = self.qdrant_service.search_similar(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Vérifier la structure de la réponse
        if isinstance(search_results, dict) and "raw_response" in search_results:
            raise ExternalServiceError(
                f"Structure de réponse Qdrant non reconnue: {search_results['raw_response']}",
                "QdrantService"
            )
        
        return search_results or []
    
    def _generate_contextual_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Génère une réponse basée sur le contexte des documents trouvés.
        """
        if not results:
            return "Je n'ai pas trouvé d'information pertinente pour répondre à votre question."
        
        # Extraire le contexte des meilleurs résultats
        context_parts = []
        has_content = False
        
        for i, result in enumerate(results[:3]):  # Utiliser les 3 meilleurs résultats
            doc_text = self.document_processor.extract_text_from_result(result)
            title = self.document_processor.extract_title_from_result(result, i)
            
            if doc_text:
                has_content = True
                # Limiter la taille du texte
                truncated_text = self.document_processor.truncate_text(
                    doc_text, 
                    self.config.max_document_length
                )
                context_parts.append(f"Document {i+1}: {title}\n{truncated_text}")
        
        if not has_content:
            return "J'ai trouvé des documents qui pourraient correspondre à votre recherche, mais je n'ai pas pu extraire leur contenu textuel."
        
        # Assembler le contexte complet
        context_text = "\n\n".join(context_parts)
        
        # Appeler le LLM pour générer une réponse
        return self._call_llm_for_answer(context_text, query)
    
    def _call_llm_for_answer(self, context_text: str, query: str) -> str:
        """
        Appelle le LLM pour générer une réponse basée sur le contexte.
        """
        try:
            # Construction des messages pour l'API Chat
            messages = [
                {"role": "system", "content": self.llm_config.general_system_prompt},
                {"role": "user", "content": f"""Voici les documents pertinents :

{context_text}

Question : {query}

Analyse ces documents et réponds à la question en te basant uniquement sur leur contenu. Cite les documents sources pour chaque information importante."""}
            ]
            
            # Appel à l'API
            response = requests.post(
                f"{self.llm_config.base_url}/api/chat",
                json={
                    "model": self.llm_config.model,
                    "messages": messages,
                    "temperature": self.llm_config.temperature,
                    "stream": False
                },
                timeout=self.llm_config.timeout
            )
            
            if not response.ok:
                raise ExternalServiceError(
                    f"Erreur API LLM: {response.status_code}",
                    "LLM",
                    response.status_code
                )
            
            # Parser la réponse
            try:
                response_json = response.json()
                answer = response_json.get("message", {}).get("content", "")
                
                if not answer:
                    return "Je n'ai pas pu générer une réponse basée sur les documents."
                
                return answer
                
            except json.JSONDecodeError:
                # Essayer d'extraire manuellement
                raw_response = response.text
                if '"content": "' in raw_response:
                    content_start = raw_response.find('"content": "') + 12
                    content_end = raw_response.find('"', content_start)
                    if content_end > content_start:
                        return raw_response[content_start:content_end]
                
                return "Je n'ai pas pu extraire correctement la réponse du LLM."
                
        except requests.Timeout:
            return "La génération de la réponse a pris trop de temps."
        except requests.RequestException as e:
            self.log(LogLevel.ERROR, f"Erreur lors de l'appel au LLM: {str(e)}")
            return "Une erreur s'est produite lors de la génération de la réponse."
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur inattendue lors de la génération: {str(e)}")
            return "Une erreur inattendue s'est produite lors de la génération de la réponse."
    
    def search_with_generate_service(
        self,
        query: str,
        discussion_id: Optional[str] = None,
        settings_id: Optional[str] = None,
        limit: int = 5,
        filters: Dict[str, Any] = None,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Effectue une recherche et génère une réponse directement à partir des résultats.

        Args:
            query: Requête de recherche textuelle
            discussion_id: ID de discussion (gardé pour compatibilité API)
            settings_id: ID de paramètres (gardé pour compatibilité API)
            limit: Nombre maximum de résultats à récupérer
            filters: Filtres à appliquer sur les métadonnées
            collection_name: Nom de la collection

        Returns:
            Dictionnaire contenant la réponse générée et les documents utilisés comme contexte
        """
        if collection_name is None:
            collection_name = self.config.default_collection
        
        def search_and_generate():
            # 1. Recherche des documents pertinents
            search_results = self.hybrid_search(
                query=query,
                limit=limit,
                filters=filters,
                use_llm_reranking=True,
                boost_keywords=True,
                generate_answer=False,
                collection_name=collection_name
            )
            
            if not search_results or not search_results.get("results"):
                return {
                    "response": "Je n'ai pas trouvé de documents pertinents pour votre requête.",
                    "documents_used": [],
                    "search_results": []
                }
            
            # 2. Génération de la réponse
            documents = search_results.get("results", [])
            answer = self._generate_contextual_answer(documents, query)
            
            # 3. Préparation des métadonnées des documents
            documents_used = []
            for i, doc in enumerate(documents[:3]):
                doc_title = self.document_processor.extract_title_from_result(doc, i)
                doc_id = doc.get("id", "") or doc.get("payload", {}).get("document_id", "")
                doc_score = doc.get("score", 0)
                
                documents_used.append({
                    "id": doc_id,
                    "title": doc_title,
                    "score": doc_score
                })
            
            return {
                "response": answer,
                "documents_used": documents_used,
                "search_results": documents
            }
        
        response = self.safe_execute(
            search_and_generate,
            f"Erreur lors de la recherche avec génération pour: {query}"
        )
        
        if response.success:
            return response.data
        else:
            return {
                "error": response.error,
                "response": f"Une erreur est survenue lors de la recherche: {response.error}",
                "documents_used": [],
                "search_results": []
            }
    
    def health_check(self) -> ServiceResponse:
        """
        Vérifie l'état de santé du service de recherche et de ses composants.
        """
        health_status = {
            "service": "SearchService",
            "status": "healthy",
            "components": {}
        }
        
        # Vérifier chaque composant
        components = [
            ("DocumentProcessor", self.document_processor),
            ("KeywordBooster", self.keyword_booster),
            ("LLMReranker", self.llm_reranker),
            ("SearchResultFilter", self.result_filter)
        ]
        
        overall_healthy = True
        
        for name, component in components:
            try:
                component_health = component.health_check()
                health_status["components"][name] = {
                    "status": "healthy" if component_health.success else "unhealthy",
                    "details": component_health.data or component_health.error
                }
                
                if not component_health.success:
                    overall_healthy = False
                    
            except Exception as e:
                health_status["components"][name] = {
                    "status": "error",
                    "details": str(e)
                }
                overall_healthy = False
        
        # Vérifier les services externes
        try:
            # Test simple d'embedding
            test_embedding = embedding_service.get_embedding("test")
            health_status["components"]["EmbeddingService"] = {
                "status": "healthy" if test_embedding else "unhealthy",
                "details": "Embedding service accessible" if test_embedding else "Embedding service inaccessible"
            }
            if not test_embedding:
                overall_healthy = False
        except Exception as e:
            health_status["components"]["EmbeddingService"] = {
                "status": "error",
                "details": str(e)
            }
            overall_healthy = False
        
        health_status["status"] = "healthy" if overall_healthy else "degraded"
        
        if overall_healthy:
            return ServiceResponse.success_response(health_status)
        else:
            return ServiceResponse.error_response("Service partiellement indisponible", "DEGRADED_SERVICE", health_status)


# Instance du service de recherche refactorisé
search_service_refactored = SearchServiceRefactored()