import logging
import requests
import json
import math
import re
import traceback
from typing import List, Dict, Any, Optional, Union

from .qdrant_service import qdrant_service, QdrantService
from .embedding_service import embedding_service
from .config_service import config_service
# Le service de génération n'est plus utilisé directement pour éviter la contamination des contextes
# from .generate_service import GenerateService
from ..libs.functions.global_functions import (
    tokenize_text, 
    remove_stopwords,
    create_sparse_vector,
    extract_entities_advanced
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt système pour llama3.1 - format simplifié et direct
LLM_SYSTEM_PROMPT = """Tu es un assistant IA spécialisé dans l'analyse de documents. Ta tâche est de :

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

LLM_RERANKER_SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans l'évaluation de la pertinence des documents. Ta tâche est d'attribuer un score de 0 à 10 à chaque document en fonction de sa pertinence par rapport à la requête.

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

class SearchService:
    """
    Service de recherche avancée combinant recherche vectorielle,
    filtrage par métadonnées et réordonnancement par LLM.
    """

    def __init__(self):
        self.collection_name = "documents"
        self.qdrant_service = QdrantService()
        # La collection "documents" est la seule à avoir le format hybride
        self.use_hybrid_search = self.collection_name == "documents"
        logger.info(f"Service de recherche initialisé avec la collection '{self.collection_name}' (hybride: {self.use_hybrid_search})")

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
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
            limit: Nombre maximum de résultats (par défaut 10)
            filters: Filtres à appliquer sur les métadonnées
            use_llm_reranking: Active le réordonnancement des résultats par LLM
            boost_keywords: Augmente le score des résultats contenant des mots-clés de la requête
            generate_answer: Génère une réponse basée sur le contexte des documents trouvés
            collection_name: Nom de la collection (par défaut self.collection_name)

        Returns:
            Dictionnaire contenant les résultats de recherche et optionnellement une réponse générée
        """
        try:
            # Basic input validation
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("Limit must be a positive integer")
            
            # Sanitize query
            query = query.strip()
            if not query:
                raise ValueError("Query cannot be empty after trimming")
                
            # Déterminer la collection à utiliser
            if collection_name is None:
                collection_name = self.collection_name
            
            # Étape 1: Recherche vectorielle initiale avec un nombre plus élevé de résultats
            initial_limit = limit * config_service.search.initial_limit_multiplier if use_llm_reranking else limit
            
            # Pour la collection "documents", on essaie d'abord la recherche hybride avancée
            search_results = None
            if collection_name == "documents":
                try:
                    logger.info(f"Essai de recherche hybride avancée pour '{collection_name}'")
                    search_results = self._hybrid_dense_sparse_search(
                        query=query,
                        limit=initial_limit,
                        filters=filters,
                        collection_name=collection_name
                    )
                except Exception as e:
                    logger.warning(f"Échec de la recherche hybride avancée: {e}, fallback vers recherche simple")
                    search_results = None
            
            # Si la recherche hybride avancée a échoué ou n'a pas été tentée, utiliser search_similar
            # qui détectera automatiquement le type de collection
            if search_results is None:
                # Obtenir le vecteur d'embedding pour la requête
                query_vector = embedding_service.get_embedding(query)
                
                if query_vector is None:
                    logger.error("Impossible de générer un embedding pour la requête")
                    return {
                        "error": "Impossible de générer un embedding pour la requête",
                        "results": [],
                        "total_found": 0,
                        "query": query
                    }
                
                # Recherche dans Qdrant (détecte automatiquement le type de collection)
                search_results = self.qdrant_service.search_similar(
                    collection_name=collection_name,
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
            
            if not search_results:
                return {"results": [], "total_found": 0, "query": query}
            
            # Étape 2: Filtres et boost mots-clés si nécessaire
            # Note: si on a utilisé la recherche hybride avancée (_hybrid_dense_sparse_search),
            # le boost des mots-clés est déjà intégré via le vecteur épars
            is_advanced_hybrid = collection_name == "documents" and search_results is not None
            
            if filters and not is_advanced_hybrid:
                search_results = self._apply_filters(search_results, filters)
                
            if boost_keywords and not is_advanced_hybrid:
                search_results = self._boost_keyword_matches(search_results, query)
            
            # Étape 3: Réordonnancement par LLM si activé
            if use_llm_reranking:
                search_results = self._rerank_with_llm(search_results, query, limit)
            else:
                # Limiter les résultats au nombre demandé
                search_results = search_results[:limit]
            
            # Étape 4: Génération d'une réponse contextuelle si demandée
            response = {
                "results": search_results,
                "total_found": len(search_results),
                "query": query
            }
            
            if generate_answer:
                answer = self._generate_answer(search_results, query)
                response["generated_answer"] = answer
            
            return response
            
        except ValueError as e:
            # Erreurs de validation - retourner une erreur structurée
            logger.warning(f"Erreur de validation: {str(e)}")
            return {"error": f"Validation error: {str(e)}", "results": [], "total_found": 0, "query": query}
        except Exception as e:
            # Autres erreurs - log complet et retour d'erreur générique
            logger.error(f"Erreur lors de la recherche: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Search error: {str(e)}", "results": [], "total_found": 0, "query": query}

    def _hybrid_dense_sparse_search(
        self, 
        query: str, 
        limit: int = 10,
        filters: Dict[str, Any] = None,
        collection_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride en combinant vecteurs denses, d'entités et épars.
        
        Args:
            query: Requête de recherche
            limit: Nombre maximum de résultats
            filters: Filtres à appliquer
            collection_name: Nom de la collection (par défaut self.collection_name)
            
        Returns:
            Liste des résultats de recherche
        """
        if collection_name is None:
            collection_name = self.collection_name
        
        # La collection est forcément hybride - nous allons directement utiliser la recherche hybride
        # et laisser la gestion des erreurs au niveau plus bas
        try:
            # 1. Générer le vecteur dense pour la requête
            dense_vector = embedding_service.get_embedding(query)
            if dense_vector is None:
                logger.error("Impossible de générer l'embedding dense pour la requête")
                return []
            
            # 2. Extraire les entités de la requête et générer leur embedding
            entities_result = extract_entities_advanced(query)
            entities = entities_result.get('entities', [])
            entity_text = ' '.join([ent['text'] for ent in entities])
            entity_vector = embedding_service.get_embedding(entity_text)
            if entity_vector is None:
                logger.warning("Impossible de générer l'embedding des entités, utilisation d'un vecteur vide")
                # Générer un vecteur vide de même taille que dense_vector
                if hasattr(dense_vector, "shape"):
                    entity_vector = [0.0] * dense_vector.shape[0]
                else:
                    entity_vector = [0.0] * len(dense_vector)
            
            # 3. Générer le vecteur épars pour la requête
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
            logger.info(f"Recherche hybride sur '{collection_name}' avec pondération: dense=0.6, entity=0.3, sparse=0.1")
            results = self.qdrant_service.hybrid_search(
                collection_name=collection_name,
                vectors=search_vectors,
                limit=limit,
                filter_conditions=filters
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche hybride dense-sparse: {e}")
            # Fallback vers la recherche standard en cas d'erreur
            logger.info(f"Fallback vers la recherche standard pour '{collection_name}'")
            query_vector = embedding_service.get_embedding(query)
            if query_vector is None:
                return []
                
            results = self.qdrant_service.search_similar(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return results

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
        
        # Extraction des entités nommées de la requête
        entities_result = extract_entities_advanced(query)
        entities = [ent['text'].lower() for ent in entities_result.get('entities', [])]
        
        # Combiner les mots-clés et les entités
        all_keywords = list(set(keywords + entities))  # Éliminer les doublons
        
        for result in results:
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "").lower()
            
            # Calcul du boost basé sur la présence des mots-clés
            keyword_matches = sum(1 for keyword in all_keywords if keyword in text)
            keyword_score = keyword_matches / len(all_keywords)
            
            # Calcul du boost basé sur la proximité des mots-clés
            proximity_score = 0
            if keyword_matches > 0:
                # Trouver la position moyenne des mots-clés
                positions = []
                for keyword in all_keywords:
                    if keyword in text:
                        pos = text.find(keyword)
                        positions.append(pos)
                if positions:
                    avg_position = sum(positions) / len(positions)
                    # Plus les mots-clés sont proches du début, plus le score est élevé
                    proximity_score = 1 - (avg_position / len(text))
            
            # Calcul du boost basé sur la densité des mots-clés
            density_score = 0
            if keyword_matches > 0:
                # Calculer la densité des mots-clés (nombre de mots-clés par 100 caractères)
                text_length = len(text)
                if text_length > 0:
                    density = (keyword_matches * 100) / text_length
                    density_score = min(1, density / 5)  # Normaliser sur une densité maximale de 5%
            
            # Combiner les différents scores avec pondération
            final_boost = (
                keyword_score * 0.5 +      # 50% pour la présence des mots-clés
                proximity_score * 0.3 +     # 30% pour la proximité
                density_score * 0.2         # 20% pour la densité
            )
            
            # Application du boost (utilise le maximum configuré)
            boost = 1 + (final_boost * config_service.search.keyword_boost_max)
            result["score"] = result.get("score", 0) * boost
            
            # Ajouter l'information sur le boost pour le débogage
            result["keyword_boost"] = {
                "boost": boost,
                "keyword_score": keyword_score,
                "proximity_score": proximity_score,
                "density_score": density_score
            }
        
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
        
        logger.info(f"Réordonnancement LLM avec {len(results)} résultats pour la requête: {query}")
        logger.info(f"URL API LLM: {config_service.llm.base_url}/api/chat")
        
        for i, result in enumerate(results):
            # Récupération du score original pour combiner plus tard
            original_score = result.get("score", 0)
            
            # Extraction du texte du document
            payload = result.get("payload", {})
            document_text = ""
            
            # Vérifier d'abord si le document a directement un champ "cleaned_text"
            if "cleaned_text" in result and isinstance(result["cleaned_text"], str) and result["cleaned_text"].strip():
                document_text = result["cleaned_text"].strip()
            
            # Sinon chercher dans le payload
            if not document_text and isinstance(payload, dict):
                document_text = payload.get("cleaned_text", "")
            
            # Si toujours vide, essayer d'autres champs courants
            if not document_text:
                for field in ["text", "content", "body"]:
                    if field in payload and isinstance(payload[field], str) and payload[field].strip():
                        document_text = payload[field].strip()
                        break
            
            # Document sans texte - conserver le score original
            if not document_text:
                logger.warning(f"Document {i+1}/{len(results)} sans texte exploitable - conservation du score original")
                reranked_results.append(result)
                continue
            
            # Tronquer le texte pour l'envoi au LLM
            max_length = config_service.search.max_document_length
            if len(document_text) > max_length:
                document_text = document_text[:max_length] + "..."
            
            # Construire le prompt pour le LLM
            prompt = f"Requête: {query}\n\nDocument: {document_text}\n\nScore de pertinence (0-10):"
            
            try:
                logger.info(f"Évaluation du document {i+1}/{len(results)} (ID: {result.get('id', 'unknown')})")
                
                # Appel au LLM via l'API de chat
                response = requests.post(
                    f"{config_service.llm.base_url}/api/chat",
                    json={
                        "model": config_service.llm.model,
                        "messages": [
                            {"role": "system", "content": LLM_RERANKER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": config_service.llm.temperature
                    }
                )
                
                # Lire la réponse brute pour le débogage
                raw_response = response.text
                logger.info(f"Réponse brute de l'API: {raw_response[:200]}...")
                
                # Essayer différentes approches pour parser la réponse JSON
                try:
                    # Méthode standard
                    response_json = response.json()
                except json.JSONDecodeError as e:
                    logger.warning(f"Erreur de parsing JSON: {e}")
                    
                    try:
                        # Essayer de nettoyer la réponse - parfois il y a des caractères en trop
                        # ou des objets JSON séparés par des sauts de ligne
                        cleaned_response = raw_response.strip()
                        
                        # Si la réponse contient plusieurs objets JSON, prendre le premier
                        if '\n' in cleaned_response:
                            cleaned_response = cleaned_response.split('\n')[0]
                        
                        response_json = json.loads(cleaned_response)
                        logger.info("Réponse JSON récupérée après nettoyage")
                    except json.JSONDecodeError:
                        logger.error("Impossible de parser la réponse JSON même après nettoyage")
                        # Extraire directement une réponse de la réponse brute si possible
                        if '"content": "' in raw_response:
                            content_start = raw_response.find('"content": "') + 12
                            content_end = raw_response.find('"', content_start)
                            if content_end > content_start:
                                extracted_text = raw_response[content_start:content_end]
                                logger.info(f"Réponse extraite directement: {extracted_text[:50]}...")
                                return extracted_text
                        
                        # Dernier recours: message d'erreur générique
                        return "Je n'ai pas pu générer une réponse basée sur les documents en raison d'un problème technique."
                
                # Extraire le score de pertinence
                llm_response = response_json.get("message", {}).get("content", "0")
                logger.info(f"Réponse extraite du LLM: '{llm_response}'")
                
                try:
                    # Extraction du score numérique
                    # D'abord essayer d'extraire uniquement les chiffres
                    digits = re.findall(r'\d+', llm_response)
                    if digits:
                        relevance_score = int(digits[0])
                    else:
                        relevance_score = int(llm_response.strip())
                    
                    # Normalisation du score dans les limites 0-10
                    if relevance_score < 0:
                        relevance_score = 0
                    elif relevance_score > 10:
                        relevance_score = 10
                        
                    logger.info(f"Score LLM: {relevance_score}/10")
                    
                    # Score combiné: utilise les poids configurés
                    normalized_original = min(10, original_score * 10)
                    combined_score = (relevance_score * config_service.search.llm_score_weight) + (normalized_original * config_service.search.original_score_weight)
                    
                    # Ajustement du score en fonction de la présence de mots-clés
                    if "keyword_boost" in result:
                        keyword_boost = result["keyword_boost"]
                        if isinstance(keyword_boost, dict):
                            # Utiliser le score de proximité pour ajuster le score final
                            proximity_factor = keyword_boost.get("proximity_score", 0)
                            combined_score = combined_score * (1 + (proximity_factor * config_service.search.proximity_adjustment_max))
                    
                    # Créer une copie du résultat original et ajouter les nouveaux scores
                    new_result = result.copy()
                    new_result["llm_score"] = relevance_score
                    new_result["score"] = combined_score
                    
                    reranked_results.append(new_result)
                    
                except ValueError as e:
                    logger.warning(f"Impossible de convertir la réponse LLM '{llm_response}' en score numérique: {e}")
                    # En cas d'erreur, conserver le résultat original
                    reranked_results.append(result)
                
            except Exception as e:
                logger.error(f"Erreur lors du réordonnancement LLM: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                # En cas d'erreur, conserver le résultat original
                reranked_results.append(result)
        
        # Réordonner les résultats selon le score combiné
        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Réordonnancement terminé: {len(reranked_results)} résultats triés")
        
        # Limiter au nombre demandé
        return reranked_results[:limit]

    def _generate_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Génère une réponse basée sur le contexte des documents trouvés.
        Utilise directement le LLM avec des paramètres optimisés pour llama3.1.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            
        Returns:
            Réponse générée
        """
        if not results:
            return "Je n'ai pas trouvé d'information pertinente pour répondre à votre question."
        
        # 1. Extraction du texte des documents
        context_parts = []
        has_content = False
        
        for i, result in enumerate(results[:3]):  # Utiliser les 3 meilleurs résultats
            # Extraction du texte du document
            doc_text = ""
            payload = result.get("payload", {})
            
            # Vérifier d'abord si le document a directement un champ "cleaned_text"
            if "cleaned_text" in result and isinstance(result["cleaned_text"], str) and result["cleaned_text"].strip():
                doc_text = result["cleaned_text"].strip()
                logger.info(f"Document {i+1} - Texte trouvé dans 'cleaned_text': {len(doc_text)} caractères")
            
            # Chercher dans le payload
            if not doc_text and isinstance(payload, dict):
                text_fields = ["cleaned_text", "text", "content", "body", "full_text"]
                for field in text_fields:
                    if field in payload and isinstance(payload[field], str) and payload[field].strip():
                        doc_text = payload[field].strip()
                        logger.info(f"Document {i+1} - Texte trouvé dans '{field}': {len(doc_text)} caractères")
                        break
                
                # Chercher dans les métadonnées
                if not doc_text and "metadata" in payload and isinstance(payload["metadata"], dict):
                    metadata = payload["metadata"]
                    for field in text_fields:
                        if field in metadata and isinstance(metadata[field], str) and metadata[field].strip():
                            doc_text = metadata[field].strip()
                            logger.info(f"Document {i+1} - Texte trouvé dans metadata.{field}: {len(doc_text)} caractères")
                            break
            
            # Déterminer le titre du document
            title = ""
            if payload:
                title = payload.get("filename", "")
                if not title and "metadata" in payload and isinstance(payload["metadata"], dict):
                    title = payload["metadata"].get("filename", "")
            
            if not title:
                title = f"Document {i+1}"
            
            # Formater le document pour le contexte
            if doc_text:
                has_content = True
                # Limiter la taille du texte
                max_length = config_service.search.max_document_length
                if len(doc_text) > max_length:
                    doc_text = doc_text[:max_length] + "..."
                
                context_parts.append(f"Document {i+1}: {title}\n{doc_text}")
                logger.info(f"Document {i+1} ajouté au contexte ({len(doc_text)} caractères)")
            else:
                logger.warning(f"Document {i+1} ignoré (pas de texte)")
        
        # 2. Vérifier si on a du contenu exploitable
        if not has_content:
            return "J'ai trouvé des documents qui pourraient correspondre à votre recherche, mais je n'ai pas pu extraire leur contenu textuel. Veuillez vérifier le format des documents."
        
        # 3. Assembler le contexte complet
        context_text = "\n\n".join(context_parts)
        
        # 4. Préparer l'appel au LLM
        if "llama" in config_service.llm.model.lower():
            # Construction des messages pour l'API Chat
            messages = [
                {"role": "system", "content": """Tu es un assistant spécialisé dans l'analyse de documents. Ta tâche est de :
1. Analyser précisément le contenu des documents fournis
2. Répondre uniquement en te basant sur les informations présentes dans ces documents
3. Ne pas faire d'hypothèses ou de suppositions non vérifiées
4. Si une information n'est pas claire ou manquante, le dire explicitement
5. Citer les documents sources pour chaque information importante
6. Ne pas confondre les documents entre eux

Format de réponse attendu :
- Réponse directe à la question
- Citations des documents sources pour chaque information
- Précision si une information est manquante ou ambiguë"""},
                {"role": "user", "content": f"""Voici les documents pertinents :

{context_text}

Question : {query}

Analyse ces documents et réponds à la question en te basant uniquement sur leur contenu. Cite les documents sources pour chaque information importante."""}
            ]
            
            # API endpoint
            api_endpoint = f"{config_service.llm.base_url}/api/chat"
            payload = {
                "model": config_service.llm.model,
                "messages": messages,
                "temperature": config_service.llm.temperature,
                "stream": False
            }
            
            # Fonction d'extraction de la réponse
            extract_function = lambda json_data: json_data.get("message", {}).get("content", "")
        else:
            # Construction du prompt pour les modèles non-chat
            prompt = f"""Contexte de recherche :

{context_text}

Question : {query}

Réponds en te basant uniquement sur les informations des documents ci-dessus :"""
            
            # API endpoint
            api_endpoint = f"{config_service.llm.base_url}/api/generate"
            payload = {
                "model": config_service.llm.model,
                "prompt": prompt,
                "temperature": config_service.llm.temperature,
                "stream": False
            }
            
            # Fonction d'extraction de la réponse
            extract_function = lambda json_data: json_data.get("response", "")
        
        # 5. Appel au LLM et gestion des erreurs
        try:
            logger.info(f"Appel à l'API LLM: {api_endpoint}")
            response = requests.post(api_endpoint, json=payload)
            
            # Récupérer la réponse brute pour débogage
            raw_response = response.text
            logger.info(f"Réponse brute (tronquée): {raw_response[:200]}...")
            
            # Essayer de parser la réponse JSON
            try:
                response_json = response.json()
                answer = extract_function(response_json)
                if not answer:
                    logger.warning("Réponse vide après extraction")
                    answer = "Je n'ai pas pu générer une réponse basée sur les documents."
            except json.JSONDecodeError as e:
                logger.warning(f"Erreur de parsing JSON: {e}")
                
                # Essayer de récupérer une réponse directement du texte brut
                if '"content":' in raw_response:
                    content_start = raw_response.find('"content": "') + 12
                    if content_start < 12:  # Si le pattern exact n'est pas trouvé
                        content_start = raw_response.find('"content":"') + 11
                    
                    if content_start > 11:  # Si un pattern a été trouvé
                        content_end = raw_response.find('"', content_start)
                        if content_end > content_start:
                            answer = raw_response[content_start:content_end]
                            logger.info(f"Réponse extraite manuellement: {answer[:50]}...")
                        else:
                            answer = "Je n'ai pas pu extraire correctement la réponse."
                    else:
                        answer = "Je n'ai pas pu extraire correctement la réponse."
                else:
                    answer = "Je n'ai pas pu générer une réponse à cause d'un problème de format."
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel au LLM: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            answer = "Une erreur s'est produite lors de la génération de la réponse."
        
        return answer

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
        Plus de dépendance au service de génération pour éviter la contamination des contextes.

        Args:
            query: Requête de recherche textuelle
            discussion_id: Non utilisé (gardé pour compatibilité API)
            settings_id: Non utilisé (gardé pour compatibilité API)
            limit: Nombre maximum de résultats à récupérer
            filters: Filtres à appliquer sur les métadonnées
            collection_name: Nom de la collection (par défaut "documents")

        Returns:
            Dictionnaire contenant la réponse générée et les documents utilisés comme contexte
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
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
            
            # 2. Préparation du contexte à partir des documents trouvés
            documents = search_results.get("results", [])
            context_documents = []
            documents_have_content = False
            
            # Afficher un exemple de document pour le débogage
            if documents:
                logger.info(f"Structure d'un document exemple: {json.dumps(documents[0], default=str)[:500]}...")
            
            for i, doc in enumerate(documents[:3]):  # Utiliser les 3 meilleurs résultats
                # IMPORTANT: Le doc peut être soit un dictionnaire avec "payload" et "score"
                # soit directement un dictionnaire avec tous les champs (cleaned_text, etc.)
                
                # Cas 1: Structure avec "payload"
                if "payload" in doc:
                    payload = doc.get("payload", {})
                    doc_id = doc.get("id", "")
                    doc_score = doc.get("score", 0)
                # Cas 2: Structure plate
                else:
                    payload = doc  # Le document lui-même est le payload
                    doc_id = doc.get("id", "") or doc.get("document_id", "")
                    doc_score = doc.get("score", 0)
                
                # Rechercher le texte dans différents champs possibles
                doc_text = ""
                
                # Pour debug - afficher tous les champs disponibles
                logger.info(f"Document {i+1} - Champs disponibles: {list(doc.keys())}")
                if "payload" in doc:
                    logger.info(f"Document {i+1} - Champs dans payload: {list(doc['payload'].keys())}")
                else:
                    logger.info(f"Document {i+1} - Document sans payload")
                
                # Vérifier d'abord si le document a directement un champ "cleaned_text"
                if "cleaned_text" in doc and isinstance(doc["cleaned_text"], str) and doc["cleaned_text"].strip():
                    doc_text = doc["cleaned_text"].strip()
                    logger.info(f"Texte trouvé directement dans le document sous 'cleaned_text': {len(doc_text)} caractères")
                
                # Sinon chercher dans le payload ou les métadonnées
                if not doc_text:
                    # Champs directs possibles
                    text_fields = ["cleaned_text", "text", "content", "body", "full_text"]
                    for field in text_fields:
                        if field in payload and isinstance(payload[field], str) and payload[field].strip():
                            doc_text = payload[field].strip()
                            logger.info(f"Texte trouvé dans le champ '{field}' du payload: {len(doc_text)} caractères")
                            break
                    
                    # Essayer de le trouver dans les métadonnées
                    if not doc_text and "metadata" in payload and isinstance(payload["metadata"], dict):
                        metadata = payload["metadata"]
                        for field in text_fields:
                            if field in metadata and isinstance(metadata[field], str) and metadata[field].strip():
                                doc_text = metadata[field].strip()
                                logger.info(f"Texte trouvé dans metadata.{field}: {len(doc_text)} caractères")
                                break
                
                # Si toujours pas de texte, essayer de convertir l'ensemble du payload en texte
                if not doc_text:
                    # Liste des champs à ignorer dans le payload pour la conversion en texte
                    ignore_fields = ["id", "document_id", "score", "tokens", "tokens_no_stopwords", 
                                    "stemmed_tokens", "lemmatized_tokens", "upload_date"]
                    
                    # Convertir les champs pertinents en texte lisible
                    try:
                        content_parts = []
                        for key, value in payload.items():
                            if key not in ignore_fields and isinstance(value, str) and value.strip():
                                content_parts.append(f"{key}: {value}")
                        
                        if content_parts:
                            doc_text = "\n".join(content_parts)
                            logger.info(f"Créé un texte à partir des champs du payload: {len(doc_text)} caractères")
                    except Exception as e:
                        logger.warning(f"Erreur lors de la conversion du payload en texte: {e}")
                
                # Vérifier si le document a du contenu
                if doc_text:
                    documents_have_content = True
                
                # Si le document n'a pas de titre, utiliser un titre générique avec le score
                title = payload.get("filename", "")
                if not title and "metadata" in payload and isinstance(payload["metadata"], dict):
                    title = payload["metadata"].get("filename", "")
                if not title:
                    title = f"Document sans titre (score: {doc_score:.2f})"
                
                doc_info = {
                    "id": doc_id,
                    "title": title,
                    "score": doc_score,
                    "text": doc_text[:config_service.search.max_document_length] if doc_text else ""  # Limiter la taille selon la configuration
                }
                context_documents.append(doc_info)
                
                # Pour débogage, afficher le texte trouvé
                if doc_text:
                    logger.info(f"Document {i+1} - Texte extrait (premiers 100 caractères): {doc_text[:100]}...")
                else:
                    logger.warning(f"Document {i+1} - Aucun texte trouvé")
            
            # 3. Vérification du contenu des documents
            if not documents_have_content:
                # Si aucun document n'a de contenu, informer l'utilisateur
                return {
                    "response": "J'ai trouvé des documents qui pourraient correspondre à votre recherche, mais je n'ai pas pu extraire leur contenu textuel. Voici les champs disponibles dans les résultats: " + 
                              str(list(documents[0].keys()) if documents else []),
                    "documents_used": context_documents,
                    "search_results": search_results.get("results", [])
                }
            
            # 4. Construction du prompt avec le contexte
            context_parts = []
            for i, doc in enumerate(context_documents, 1):
                if doc['text']:
                    # Formater chaque document pour le prompt
                    context_parts.append(f"Document {i}: {doc['title']}\n{doc['text']}")
            
            # Assembler le contexte complet
            context_text = "\n\n".join(context_parts)
            
            # 5. Appel au LLM selon le type de modèle
            answer = ""
            
            try:
                if "llama" in config_service.llm.model.lower():
                    # Pour Llama, utiliser l'API de chat
                    logger.info(f"Utilisation de l'API chat pour le modèle {config_service.llm.model}")
                    
                    # Construction des messages pour l'API Chat
                    messages = [
                        {"role": "system", "content": """Tu es un assistant spécialisé dans l'analyse de documents. Ta tâche est de :
1. Analyser précisément le contenu des documents fournis
2. Répondre uniquement en te basant sur les informations présentes dans ces documents
3. Ne pas faire d'hypothèses ou de suppositions non vérifiées
4. Si une information n'est pas claire ou manquante, le dire explicitement
5. Citer les documents sources pour chaque information importante
6. Ne pas confondre les documents entre eux

Format de réponse attendu :
- Réponse directe à la question
- Citations des documents sources pour chaque information
- Précision si une information est manquante ou ambiguë"""},
                        {"role": "user", "content": f"""Voici les documents pertinents :

{context_text}

Question : {query}

Analyse ces documents et réponds à la question en te basant uniquement sur leur contenu. Cite les documents sources pour chaque information importante."""}
                    ]
                    
                    # Appel à l'API de chat
                    response = requests.post(
                        f"{config_service.llm.base_url}/api/chat",
                        json={
                            "model": config_service.llm.model,
                            "messages": messages,
                            "temperature": config_service.llm.temperature,
                            "stream": False
                        }
                    )
                    
                    # Lire la réponse brute pour le débogage
                    raw_response = response.text
                    logger.info(f"Réponse brute de l'API: {raw_response[:200]}...")
                    
                    # Essayer différentes approches pour parser la réponse JSON
                    try:
                        # Méthode standard
                        response_json = response.json()
                        # Extraire la réponse
                        answer = response_json.get("message", {}).get("content", "")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Erreur de parsing JSON: {e}")
                        
                        try:
                            # Essayer de nettoyer la réponse - parfois il y a des caractères en trop
                            # ou des objets JSON séparés par des sauts de ligne
                            cleaned_response = raw_response.strip()
                            
                            # Si la réponse contient plusieurs objets JSON, prendre le premier
                            if '\n' in cleaned_response:
                                cleaned_response = cleaned_response.split('\n')[0]
                            
                            response_json = json.loads(cleaned_response)
                            logger.info("Réponse JSON récupérée après nettoyage")
                            # Extraire la réponse
                            answer = response_json.get("message", {}).get("content", "")
                        except json.JSONDecodeError:
                            logger.error("Impossible de parser la réponse JSON même après nettoyage")
                            # Extraire directement une réponse de la réponse brute si possible
                            if '"content": "' in raw_response:
                                content_start = raw_response.find('"content": "') + 12
                                content_end = raw_response.find('"', content_start)
                                if content_end > content_start:
                                    answer = raw_response[content_start:content_end]
                                    logger.info(f"Réponse extraite directement: {answer[:50]}...")
                            
                            # Si toujours pas de réponse, message d'erreur
                            if not answer:
                                answer = "Je n'ai pas pu générer une réponse à cause d'un problème technique avec le LLM."
                else:
                    # Pour d'autres modèles, utiliser l'API de génération
                    logger.info(f"Utilisation de l'API generate pour le modèle {config_service.llm.model}")
                    
                    # Construction du prompt pour l'API de génération
                    prompt = f"""Contexte de recherche:

{context_text}

Question: {query}

Réponds en te basant uniquement sur les informations des documents ci-dessus:"""
                    
                    # Appel à l'API de génération
                    response = requests.post(
                        f"{config_service.llm.base_url}/api/generate",
                        json={
                            "model": config_service.llm.model,
                            "prompt": prompt,
                            "temperature": config_service.llm.temperature,
                            "stream": False
                        }
                    )
                    
                    # Lire la réponse brute pour le débogage
                    raw_response = response.text
                    logger.info(f"Réponse brute de l'API: {raw_response[:200]}...")
                    
                    # Parser la réponse
                    try:
                        response_json = response.json()
                        answer = response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.error("Impossible de parser la réponse JSON")
                        answer = "Je n'ai pas pu générer une réponse à cause d'un problème technique avec le LLM."
                
                # Si pas de réponse valide, fournir un message d'erreur
                if not answer:
                    answer = "Je n'ai pas pu générer une réponse basée sur les documents."
                
            except Exception as e:
                logger.error(f"Erreur lors de l'appel au LLM: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                answer = f"Une erreur est survenue lors de la génération de la réponse: {str(e)}"
            
            # 6. Construction de la réponse finale
            result = {
                "response": answer,
                "documents_used": context_documents,
                "search_results": search_results.get("results", [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "response": f"Une erreur est survenue lors de la recherche: {str(e)}",
                "documents_used": [],
                "search_results": []
            }


# Instance singleton du service
search_service = SearchService() 