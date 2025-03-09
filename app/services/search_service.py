import logging
import os
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import math
import re

from .qdrant_service import qdrant_service, QdrantService
from .embedding_service import embedding_service
from .generate_service import GenerateService

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration du LLM
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-coder")
LLM_RERANKER_SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans la recherche documentaire. Ta tâche est d'évaluer la pertinence des documents 
par rapport à la requête de l'utilisateur. Pour chaque document, attribue un score de pertinence de 0 à 10, 
où 10 est parfaitement pertinent et 0 est totalement non pertinent.
Tiens compte de :
1. La correspondance sémantique entre la requête et le contenu du document
2. La présence des entités nommées mentionnées dans la requête
3. La précision et l'exhaustivité des informations par rapport à la requête
Retourne uniquement un nombre entier de 0 à 10 sans autre texte ou explication.
"""


class SearchService:
    """
    Service de recherche avancée combinant recherche vectorielle,
    filtrage par métadonnées et réordonnancement par LLM.
    """

    def __init__(self):
        self.collection_name = "documents"
        self.qdrant_service = QdrantService()
        self.generate_service = GenerateService()
        logger.info(f"Service de recherche initialisé avec la collection '{self.collection_name}'")

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        filters: Dict[str, Any] = None,
        use_llm_reranking: bool = True,
        boost_keywords: bool = True,
        generate_answer: bool = False,
    ) -> Dict[str, Any]:
        """
        Effectue une recherche hybride combinant recherche vectorielle et filtrage par métadonnées,
        avec réordonnancement optionnel par LLM.

        Args:
            query: Requête de recherche textuelle
            limit: Nombre maximum de résultats
            filters: Filtres à appliquer sur les métadonnées
            use_llm_reranking: Active le réordonnancement des résultats par LLM
            boost_keywords: Augmente le score des résultats contenant des mots-clés de la requête
            generate_answer: Génère une réponse basée sur le contexte des documents trouvés

        Returns:
            Dictionnaire contenant les résultats de recherche et optionnellement une réponse générée
        """
        try:
            # Étape 1: Recherche vectorielle initiale avec un nombre plus élevé de résultats
            # (pour permettre le réordonnancement)
            initial_limit = limit * 3 if use_llm_reranking else limit
            query_vector = embedding_service.get_embedding(query)
            
            if query_vector is None:
                logger.error("Impossible de générer un embedding pour la requête")
                return {
                    "error": "Impossible de générer un embedding pour la requête",
                    "results": [],
                    "total_found": 0,
                    "query": query
                }
            
            # Recherche initiale dans Qdrant
            try:
                search_results = self.qdrant_service.search_similar(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=initial_limit
                )
                
                # Vérifier si la réponse a une structure valide
                if isinstance(search_results, dict) and "raw_response" in search_results:
                    logger.error(f"Structure de réponse Qdrant non reconnue: {search_results['raw_response']}")
                    return {
                        "error": "Erreur lors de la communication avec Qdrant",
                        "results": [],
                        "total_found": 0,
                        "query": query
                    }
                
            except Exception as e:
                logger.error(f"Erreur lors de la recherche initiale: {e}")
                return {
                    "error": f"Erreur lors de la recherche: {str(e)}",
                    "results": [],
                    "total_found": 0,
                    "query": query
                }
            
            if not search_results:
                return {"results": [], "total_found": 0, "query": query}
            
            # Étape 2: Appliquer des filtres si spécifiés
            if filters:
                search_results = self._apply_filters(search_results, filters)
            
            # Étape 3: Boost des résultats contenant des mots-clés de la requête
            if boost_keywords:
                search_results = self._boost_keyword_matches(search_results, query)
            
            # Étape 4: Réordonnancement par LLM si activé
            if use_llm_reranking:
                search_results = self._rerank_with_llm(search_results, query, limit)
            else:
                # Limiter les résultats au nombre demandé
                search_results = search_results[:limit]
            
            # Étape 5: Génération d'une réponse contextuelle si demandée
            response = {
                "results": search_results,
                "total_found": len(search_results),
                "query": query
            }
            
            if generate_answer:
                answer = self._generate_answer(search_results, query)
                response["generated_answer"] = answer
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}")
            return {"error": str(e), "results": [], "total_found": 0}

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applique des filtres sur les résultats de recherche.
        
        Args:
            results: Liste des résultats de recherche
            filters: Dictionnaire de filtres à appliquer
        
        Returns:
            Liste filtrée des résultats
        """
        filtered_results = []
        
        for result in results:
            match = True
            payload = result.get("payload", {})
            
            for key, value in filters.items():
                # Gestion des filtres de date (intervalle)
                if key == "upload_date_range" and isinstance(value, dict):
                    date = payload.get("upload_date", "")
                    if "start" in value and date < value["start"]:
                        match = False
                        break
                    if "end" in value and date > value["end"]:
                        match = False
                        break
                # Gestion des filtres de liste (contient au moins un élément)
                elif key.startswith("has_") and isinstance(value, bool) and value:
                    list_name = key[4:]  # Retirer le préfixe "has_"
                    list_items = payload.get(list_name, [])
                    if not list_items:
                        match = False
                        break
                # Filtre exact
                elif payload.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered_results.append(result)
        
        return filtered_results

    def _boost_keyword_matches(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Augmente le score des résultats contenant des mots-clés de la requête.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
        
        Returns:
            Liste des résultats avec scores ajustés
        """
        # Extraction des mots-clés significatifs (plus de 3 caractères)
        keywords = [word.lower() for word in re.findall(r'\b\w{3,}\b', query)]
        if not keywords:
            return results
        
        for result in results:
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "").lower()
            
            # Calcul du boost basé sur la présence des mots-clés
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            keyword_score = keyword_matches / len(keywords)
            
            # Application du boost (maximum 30% d'augmentation)
            boost = 1 + (keyword_score * 0.3)
            result["score"] = result.get("score", 0) * boost
            
            # Ajouter l'information sur le boost pour le débogage
            result["keyword_boost"] = boost
        
        # Réordonner les résultats selon le nouveau score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    def _rerank_with_llm(self, results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Réordonne les résultats en utilisant un LLM pour évaluer leur pertinence.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            limit: Nombre maximum de résultats à retourner
            
        Returns:
            Liste réordonnée des résultats
        """
        if not results:
            return []
        
        reranked_results = []
        
        for result in results:
            payload = result.get("payload", {})
            document_text = payload.get("cleaned_text", "")
            
            # Tronquer le texte pour l'envoi au LLM
            if len(document_text) > 2000:
                document_text = document_text[:2000] + "..."
            
            # Construire le prompt pour le LLM
            prompt = f"Requête: {query}\n\nDocument: {document_text}\n\nScore de pertinence (0-10):"
            
            try:
                # Appel au LLM
                response = requests.post(
                    f"{LLM_BASE_URL}/api/chat",
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {"role": "system", "content": LLM_RERANKER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1  # Faible température pour des résultats plus cohérents
                    }
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                # Extraire le score de pertinence
                llm_response = response_json.get("message", {}).get("content", "0")
                try:
                    relevance_score = int(llm_response.strip())
                    if relevance_score < 0:
                        relevance_score = 0
                    elif relevance_score > 10:
                        relevance_score = 10
                except ValueError:
                    relevance_score = 0
                    
                # Normaliser le score pour être combiné avec le score initial
                normalized_score = relevance_score / 10
                
                # Combiner les scores (70% LLM, 30% similarité vectorielle)
                original_score = result.get("score", 0)
                combined_score = (normalized_score * 0.7) + (original_score * 0.3)
                
                # Créer un nouveau résultat avec le score combiné
                new_result = result.copy()
                new_result["original_score"] = original_score
                new_result["llm_score"] = relevance_score
                new_result["score"] = combined_score
                
                reranked_results.append(new_result)
                
            except Exception as e:
                logger.error(f"Erreur lors du réordonnancement LLM: {str(e)}")
                # En cas d'erreur, conserver le résultat original
                reranked_results.append(result)
        
        # Réordonner les résultats selon le score combiné
        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Limiter au nombre demandé
        return reranked_results[:limit]

    def _generate_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Génère une réponse basée sur le contexte des documents trouvés.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            
        Returns:
            Réponse générée
        """
        if not results:
            return "Je n'ai pas trouvé d'information pertinente pour répondre à votre question."
        
        # Extraire les textes des documents les plus pertinents
        contexts = []
        for result in results[:3]:  # Utiliser les 3 meilleurs résultats
            payload = result.get("payload", {})
            document_text = payload.get("cleaned_text", "")
            
            # Limiter la taille du contexte
            if len(document_text) > 1000:
                document_text = document_text[:1000] + "..."
                
            contexts.append(document_text)
        
        # Construire le contexte complet
        full_context = "\n\n---\n\n".join(contexts)
        
        # Construire le prompt pour le LLM
        system_prompt = """
        Tu es un assistant IA expert dans la recherche d'informations. Ton rôle est de répondre aux questions 
        en te basant uniquement sur les contextes fournis. Si l'information n'est pas dans les contextes, 
        indique clairement que tu ne peux pas répondre à la question avec les informations disponibles.
        Ne jamais inventer d'information. Cite les sources dans ta réponse quand c'est pertinent.
        """
        
        user_prompt = f"Contextes:\n\n{full_context}\n\nQuestion: {query}\n\nRéponds à la question en te basant uniquement sur les contextes fournis."
        
        try:
            # Appel au LLM
            response = requests.post(
                f"{LLM_BASE_URL}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3  # Température modérée pour équilibrer précision et fluidité
                }
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            # Extraire la réponse générée
            answer = response_json.get("message", {}).get("content", "")
            if not answer:
                return "Je n'ai pas pu générer une réponse basée sur les documents trouvés."
                
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {str(e)}")
            return f"Une erreur s'est produite lors de la génération de la réponse: {str(e)}"

    def search_with_generate_service(
        self,
        query: str,
        discussion_id: Optional[str] = None,
        settings_id: Optional[str] = None,
        limit: int = 5,
        filters: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Effectue une recherche interne en utilisant le service de génération existant.
        Cette fonction récupère les documents pertinents, puis les transmet au LLM via le
        service de génération pour obtenir une réponse contextualisée.

        Args:
            query: Requête de recherche textuelle
            discussion_id: ID de la discussion (optionnel)
            settings_id: ID des paramètres à utiliser (optionnel)
            limit: Nombre maximum de résultats à récupérer
            filters: Filtres à appliquer sur les métadonnées

        Returns:
            Dictionnaire contenant la réponse générée et les documents utilisés comme contexte
        """
        try:
            # 1. Recherche des documents pertinents avec notre service de recherche
            search_results = self.hybrid_search(
                query=query,
                limit=limit,
                filters=filters,
                use_llm_reranking=True,
                boost_keywords=True,
                generate_answer=False  # On ne génère pas de réponse ici, on le fera avec generate_service
            )
            
            if not search_results or not search_results.get("results"):
                return {
                    "response": "Je n'ai pas trouvé de documents pertinents pour votre requête.",
                    "documents_used": [],
                    "search_results": []
                }
            
            # 2. Préparation du contexte à partir des documents trouvés
            documents = search_results.get("results", [])
            context_documents = []
            
            for doc in documents[:3]:  # Utiliser les 3 meilleurs résultats
                payload = doc.get("payload", {})
                doc_info = {
                    "id": doc.get("id", ""),
                    "title": payload.get("filename", "Document sans titre"),
                    "score": doc.get("score", 0),
                    "text": payload.get("cleaned_text", "")[:1000]  # Limiter la taille
                }
                context_documents.append(doc_info)
            
            # 3. Formater le contexte pour le service de génération
            context_text = "\n\n--- Contexte de recherche ---\n"
            for i, doc in enumerate(context_documents, 1):
                context_text += f"\nDocument {i}: {doc['title']}\n"
                context_text += f"{doc['text']}\n"
                context_text += "---\n"
                
            # 4. Préparer le prompt pour le LLM (pour informer qu'il s'agit d'une recherche interne)
            search_instruction = (
                "Je vous ai fourni des extraits de documents pertinents pour votre requête. "
                "Utilisez ces informations pour répondre à ma question: "
            )
            
            # 5. Appel au service de génération existant avec le contexte
            enhanced_query = search_instruction + query
            
            generate_response = self.generate_service.generate_response(
                discussion_id=discussion_id,
                settings_id=settings_id,
                current_message=enhanced_query,
                additional_info=context_text
            )
            
            # 6. Ajout des documents utilisés à la réponse pour pouvoir les afficher dans l'UI
            generate_response["documents_used"] = context_documents
            generate_response["search_results"] = search_results.get("results", [])
            
            return generate_response
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche avec generate_service: {str(e)}")
            return {
                "error": str(e),
                "response": "Une erreur est survenue lors de la recherche dans les documents internes.",
                "documents_used": [],
                "search_results": []
            }


# Instance singleton du service
search_service = SearchService() 