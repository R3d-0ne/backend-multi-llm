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

    def _preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Prétraite une requête pour améliorer la recherche.
        Inclut l'expansion de requête, la reformulation et l'extraction d'entités.
        
        Args:
            query: Requête originale
            
        Returns:
            Dictionnaire contenant la requête originale, les variantes et les entités
        """
        if not query or not isinstance(query, str):
            return {"original": query, "variants": [query], "entities": []}
        
        query = query.strip()
        
        # 1. Extraction des entités
        entities_result = extract_entities_advanced(query)
        entities = entities_result.get('entities', [])
        
        # 2. Création de variantes de la requête
        variants = [query]  # Commencer avec la requête originale
        
        # Ajouter des variantes basées sur les entités
        if entities:
            entity_texts = [ent['text'] for ent in entities if ent.get('score', 0) > 0.7]
            if entity_texts:
                # Variante avec seulement les entités importantes
                entity_query = ' '.join(entity_texts)
                if entity_query != query:
                    variants.append(entity_query)
        
        # 3. Expansion avec des synonymes simples (pour le français)
        synonyms_map = {
            'comment': ['comment faire', 'comment procéder', 'comment obtenir'],
            'obtenir': ['avoir', 'récupérer', 'télécharger'],
            'demande': ['requête', 'demander', 'solliciter'],
            'document': ['papier', 'formulaire', 'certificat'],
            'aide': ['assistance', 'support', 'aide financière'],
            'logement': ['habitation', 'appartement', 'maison'],
            'travail': ['emploi', 'emploi', 'carrière'],
            'santé': ['médical', 'soins', 'assurance maladie'],
            'famille': ['enfants', 'parental', 'maternité'],
            'retraite': ['pension', 'retraite', 'caisse de retraite'],
            # Ajout de synonymes spécifiques pour les documents financiers
            'rib': ['relevé d\'identité bancaire', 'identité bancaire', 'coordonnées bancaires', 'iban', 'bic', 'banque'],
            'facture': ['note', 'devis', 'avoir', 'facturation'],
            'compte': ['compte bancaire', 'compte courant', 'compte épargne'],
            'banque': ['établissement bancaire', 'institution financière', 'crédit', 'caisse'],
            # Ajout de synonymes pour la sécurité sociale
            'sécurité': ['securite', 'sociale', 'caf', 'cpam', 'assurance', 'maladie', 'retraite', 'pension'],
            'sociale': ['sécurité', 'securite', 'caf', 'cpam', 'assurance', 'maladie', 'retraite', 'pension'],
            'document': ['papier', 'formulaire', 'certificat', 'attestation', 'justificatif']
        }
        
        # Ajouter des variantes avec synonymes
        words = query.lower().split()
        for word in words:
            if word in synonyms_map:
                for synonym in synonyms_map[word][:2]:  # Limiter à 2 synonymes
                    variant = query.replace(word, synonym)
                    if variant not in variants:
                        variants.append(variant)
        
        # 4. Nettoyage et déduplication
        variants = list(dict.fromkeys(variants))  # Supprimer les doublons en gardant l'ordre
        
        logger.info(f"Requête prétraitée: {len(variants)} variantes générées, {len(entities)} entités extraites")
        
        return {
            "original": query,
            "variants": variants,
            "entities": entities,
            "main_entities": [ent['text'] for ent in entities if ent.get('score', 0) > 0.7]
        }

    def _calculate_retrieval_metrics(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Calcule des métriques de qualité pour évaluer la pertinence des résultats.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête originale
            
        Returns:
            Dictionnaire contenant les métriques calculées
        """
        if not results:
            return {
                "avg_score": 0.0,
                "score_std": 0.0,
                "high_quality_results": 0,
                "coverage_score": 0.0,
                "diversity_score": 0.0
            }
        
        # 1. Métriques de score
        scores = [result.get("score", 0) for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        score_std = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0.0
        
        # 2. Nombre de résultats de haute qualité (score > 0.7)
        high_quality_results = sum(1 for score in scores if score > 0.7)
        
        # 3. Score de couverture (présence de mots-clés de la requête)
        query_words = set(query.lower().split())
        coverage_scores = []
        for result in results:
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "").lower()
            found_words = sum(1 for word in query_words if word in text)
            coverage = found_words / len(query_words) if query_words else 0.0
            coverage_scores.append(coverage)
        
        coverage_score = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
        
        # 4. Score de diversité (basé sur la variété des sources/fichiers)
        sources = set()
        for result in results:
            payload = result.get("payload", {})
            source = payload.get("filename", payload.get("source", "unknown"))
            sources.add(source)
        
        diversity_score = len(sources) / len(results) if results else 0.0
        
        # 5. Score de cohérence (similarité entre les scores)
        coherence_score = 1.0 - (score_std / (avg_score + 0.001))  # Éviter division par zéro
        
        return {
            "avg_score": round(avg_score, 3),
            "score_std": round(score_std, 3),
            "high_quality_results": high_quality_results,
            "coverage_score": round(coverage_score, 3),
            "diversity_score": round(diversity_score, 3),
            "coherence_score": round(coherence_score, 3),
            "total_sources": len(sources)
        }

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
            
            # Prétraitement de la requête pour améliorer la recherche
            query_info = self._preprocess_query(query)
            original_query = query_info["original"]
            query_variants = query_info["variants"]
            entities = query_info["entities"]
                
            # Déterminer la collection à utiliser
            if collection_name is None:
                collection_name = self.collection_name
            
            # Étape 1: Recherche vectorielle initiale avec un nombre plus élevé de résultats
            initial_limit = limit * config_service.search.initial_limit_multiplier if use_llm_reranking else limit

            # Utiliser la requête principale (première variante) pour la recherche
            main_query = query_variants[0] if query_variants else original_query
            
            # Pour la collection "documents", on essaie d'abord la recherche hybride avancée
            search_results = None
            if collection_name == "documents":
                try:
                    logger.info(f"Essai de recherche hybride avancée pour '{collection_name}' avec requête: '{main_query}'")
                    search_results = self._hybrid_dense_sparse_search(
                        query=main_query,
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
                # Obtenir le vecteur d'embedding pour la requête principale
                query_vector = embedding_service.get_embedding(main_query)

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
                search_results = self._boost_keyword_matches(search_results, original_query)

            # Étape 3: Réordonnancement par LLM si activé
            if use_llm_reranking:
                search_results = self._rerank_with_llm(search_results, original_query, limit)
            else:
                # Limiter les résultats au nombre demandé
                search_results = search_results[:limit]

            # Étape 4: Calcul des métriques de qualité
            metrics = self._calculate_retrieval_metrics(search_results, original_query)
            
            # Étape 5: Génération d'une réponse contextuelle si demandée
            response = {
                "results": search_results,
                "total_found": len(search_results),
                "query": original_query,
                "query_info": {
                    "original": original_query,
                    "variants_count": len(query_variants),
                    "entities_count": len(entities),
                    "main_entities": query_info.get("main_entities", [])
                },
                "metrics": metrics
            }
            
            if generate_answer:
                answer = self._generate_answer(search_results, original_query)
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
        Recherche hybride : dense, entity, sparse, fusion des scores côté Python.
        Utilise les vecteurs du champ 'vectors' retourné par Qdrant.
        Prend en compte les champs llm_keywords et llm_entities du document.
        """
        from ..libs.functions.global_functions import (
            compute_embedding_similarity_from_vectors,
            create_sparse_vector,
            tokenize_text,
            remove_stopwords
        )
        import numpy as np

        if collection_name is None:
            collection_name = self.collection_name
        # 1. Embedding dense de la requête
        dense_vector = embedding_service.get_embedding(query)
        if dense_vector is None:
            logger.error("Impossible de générer l'embedding dense pour la requête")
            return []
        # 2. Embedding entité de la requête
        entities_result = extract_entities_advanced(query)
        entities = entities_result.get('entities', [])
        entity_texts = [ent['text'] for ent in entities]
        entity_text = ' '.join(entity_texts)
        entity_vector = embedding_service.get_embedding(entity_text) if entity_text else [0.0] * len(dense_vector)
        # 3. Sparse vector de la requête
        tokens = tokenize_text(query)
        tokens_no_stopwords = remove_stopwords(tokens)
        sparse_vector = create_sparse_vector(tokens_no_stopwords, vocab_size=10000)
        # 4. Recherche initiale Qdrant (dense uniquement)
        results = self.qdrant_service.search_similar(
            collection_name=collection_name,
            query_vector=dense_vector,
            limit=limit*3,  # On élargit le pool pour un meilleur rerank
            filters=filters
        )
        if not results:
            return []
        reranked = []
        for doc in results:
            payload = doc.get("payload", {})
            vectors = doc.get("vectors", {})
            doc_dense = vectors.get("dense") or doc.get("vector") or payload.get("dense_vector")
            doc_entity = vectors.get("entity") or payload.get("entity_vector")
            doc_sparse = vectors.get("sparse") or payload.get("sparse_vector")
            # Champs LLM
            llm_keywords = payload.get("llm_keywords", [])
            llm_entities = payload.get("llm_entities", [])
            # Vecteur sparse enrichi (texte + llm_keywords)
            doc_tokens = []
            if isinstance(payload.get("cleaned_text"), str):
                doc_tokens = tokenize_text(payload["cleaned_text"])
            doc_tokens_no_stopwords = remove_stopwords(doc_tokens)
            doc_tokens_full = doc_tokens_no_stopwords + [kw.lower() for kw in llm_keywords if isinstance(kw, str)]
            doc_sparse_llm = create_sparse_vector(doc_tokens_full, vocab_size=10000) if doc_tokens_full else doc_sparse
            # Vecteur entité enrichi (entités extraites + llm_entities)
            doc_entities_texts = []
            if isinstance(payload.get("named_entities_bert"), list):
                doc_entities_texts += [ent for ent in payload["named_entities_bert"] if isinstance(ent, str)]
            doc_entities_texts += [ent["text"] for ent in llm_entities if isinstance(ent, dict) and "text" in ent]
            doc_entities_text = ' '.join(doc_entities_texts)
            doc_entity_llm = embedding_service.get_embedding(doc_entities_text) if doc_entities_text else doc_entity
            # Calcul des similarités
            try:
                dense_score = compute_embedding_similarity_from_vectors(
                    np.array(dense_vector), np.array(doc_dense)) if doc_dense is not None else 0.0
            except Exception:
                dense_score = 0.0
            try:
                entity_score = compute_embedding_similarity_from_vectors(
                    np.array(entity_vector), np.array(doc_entity_llm)) if doc_entity_llm is not None else 0.0
                # Bonus si une entité de la requête est présente dans les llm_entities du doc
                bonus = 0.0
                doc_llm_entities_texts = [ent["text"].lower() for ent in llm_entities if isinstance(ent, dict) and "text" in ent]
                for ent in entity_texts:
                    if ent.lower() in doc_llm_entities_texts:
                        bonus = 0.05
                        break
                entity_score += bonus
            except Exception:
                entity_score = 0.0
            try:
                sparse_score = compute_embedding_similarity_from_vectors(
                    np.array(sparse_vector), np.array(doc_sparse_llm)) if doc_sparse_llm is not None else 0.0
            except Exception:
                sparse_score = 0.0
            # Poids optimisés basés sur l'analyse des performances
            # Dense: 70% (sémantique principal), Entity: 20% (entités importantes), Sparse: 10% (mots-clés exacts)
            combined_score = 0.7 * dense_score + 0.2 * entity_score + 0.1 * sparse_score
            doc_result = {
                **doc,
                "dense_score": dense_score,
                "entity_score": entity_score,
                "sparse_score": sparse_score,
                "combined_score": combined_score,
                "llm_keywords": llm_keywords,
                "llm_entities": llm_entities
            }
            reranked.append(doc_result)
        reranked.sort(key=lambda x: x["combined_score"], reverse=True)
        return reranked[:limit]

    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applique des filtres sur les résultats de recherche selon les métadonnées.
        Args:
            results: Liste des résultats de recherche
            filters: Dictionnaire de filtres à appliquer
        Returns:
            Liste filtrée des résultats
        """
        filtered_results = []
        for result in results:
            payload = result.get("payload", {})
            match = True
            for key, value in filters.items():
                # Filtre sur une plage de dates
                if key == "upload_date_range" and isinstance(value, dict):
                    date = payload.get("upload_date", "")
                    if "start" in value and date < value["start"]:
                        match = False
                        break
                    if "end" in value and date > value["end"]:
                        match = False
                        break
                # Filtre booléen sur la présence d'une liste non vide
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
        Version améliorée avec calcul de proximité et densité des mots-clés.
        """
        keywords = [word.lower() for word in re.findall(r'\b\w{3,}\b', query)]
        query_lower = query.lower().strip()
        
        # Utiliser les mots de la requête comme mots-clés principaux
        query_words = query_lower.split()
        
        for result in results:
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "").lower()
            title = payload.get("filename", "").lower()
            
            # Boost fort si la requête exacte est dans le texte ou le titre
            exact_match = False
            if query_lower in text or query_lower in title:
                exact_match = True
            
            # Calcul du boost classique basé sur les mots-clés
            keyword_matches = sum(1 for keyword in keywords if keyword in text)
            keyword_score = keyword_matches / len(keywords) if keywords else 0
            
            # Boost basé sur la correspondance des mots de la requête
            document_type_boost = 1.0
            query_matches = sum(1 for word in query_words if word in text or word in title)
            if query_matches > 0:
                document_type_boost = 1.0 + (query_matches * 0.2)  # Boost progressif basé sur les mots de la requête
            
            # Calcul de la proximité des mots-clés (distance moyenne entre les mots-clés)
            proximity_score = 0.0
            if len(keywords) > 1:
                positions = []
                for keyword in keywords:
                    pos = text.find(keyword)
                    if pos != -1:
                        positions.append(pos)
                
                if len(positions) > 1:
                    # Calculer la distance moyenne entre les positions
                    distances = [abs(positions[i] - positions[i-1]) for i in range(1, len(positions))]
                    avg_distance = sum(distances) / len(distances)
                    # Score de proximité inversement proportionnel à la distance
                    proximity_score = max(0, 1 - (avg_distance / 1000))  # Normaliser sur 1000 caractères
            
            # Calcul de la densité des mots-clés
            total_words = len(text.split())
            keyword_density = keyword_matches / total_words if total_words > 0 else 0
            density_score = min(1.0, keyword_density * 10)  # Normaliser
            
            # Calcul du boost final
            if exact_match:
                boost = 2.5  # Boost très fort pour correspondance exacte
            else:
                # Boost progressif basé sur les différents scores
                boost = 1.0 + (keyword_score * 0.5) + (proximity_score * 0.3) + (density_score * 0.2)
                boost = min(boost, 2.0)  # Limiter le boost maximum
            
            # Appliquer le boost de type de document
            boost *= document_type_boost
            
            # Appliquer le boost au score
            result["score"] = result.get("score", 0) * boost

            # Ajouter l'information sur le boost pour le débogage
            result["keyword_boost"] = {
                "boost": boost,
                "keyword_score": keyword_score,
                "proximity_score": proximity_score,
                "density_score": density_score,
                "exact_match": exact_match,
                "document_type_boost": document_type_boost
            }

        # Réordonner les résultats selon le nouveau score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    def _generate_intelligent_fallback_scores(self, documents_with_text: List[Dict[str, Any]], query: str) -> List[int]:
        """
        Génère des scores intelligents basés sur l'analyse du contenu des documents
        quand le LLM ne retourne pas de scores valides.
        
        Args:
            documents_with_text: Liste des documents avec leur texte
            query: Requête de recherche
            
        Returns:
            Liste des scores pour chaque document
        """
        query_lower = query.lower()
        scores = []
        
        # Analyser la requête pour extraire les mots-clés pertinents
        query_words = query_lower.split()
        
        # Mots-clés pour différents types de documents (plus génériques)
        document_type_keywords = {
            'bancaire': ['rib', 'relevé', 'identité', 'bancaire', 'iban', 'bic', 'banque', 'compte', 'coordonnées'],
            'facture': ['facture', 'note', 'devis', 'avoir', 'facturation', 'montant', 'euros', '€', 'prix', 'total'],
            'sécurité_sociale': ['sécurité', 'sociale', 'securite', 'caf', 'cpam', 'assurance', 'maladie', 'retraite', 'pension', 'social'],
            'général': query_words  # Utiliser les mots de la requête comme mots-clés généraux
        }
        
        # Déterminer le type de document recherché de manière plus flexible
        detected_types = []
        for doc_type, keywords in document_type_keywords.items():
            if any(word in query_lower for word in keywords):
                detected_types.append(doc_type)
        
        for i, doc in enumerate(documents_with_text):
            text = doc['text'].lower()
            filename = doc['result'].get('payload', {}).get('filename', '').lower()
            original_score = doc['result'].get('score', 0)
            
            # Score de base basé sur la position (les premiers sont généralement plus pertinents)
            base_score = max(3, 8 - i)
            
            # Boost basé sur le contenu du document
            content_boost = 0
            
            for doc_type in detected_types:
                if doc_type == 'général':
                    # Pour les mots-clés généraux, boost modéré
                    matches = sum(1 for word in query_words if word in text or word in filename)
                    content_boost += min(2, matches * 0.5)
                else:
                    # Pour les types spécifiques, boost plus fort
                    keywords = document_type_keywords[doc_type]
                    matches = sum(1 for word in keywords if word in text or word in filename)
                    if matches > 0:
                        content_boost += min(3, matches)  # Boost jusqu'à 3 points
                        logger.info(f"Document {i+1}: {matches} mots-clés {doc_type} trouvés, boost: +{min(3, matches)}")
            
            # Si aucun type spécifique détecté, utiliser une recherche générale
            if not detected_types or detected_types == ['général']:
                matches = sum(1 for word in query_words if word in text or word in filename)
                content_boost = min(2, matches)
            
            # Score final
            final_score = max(1, min(10, base_score + content_boost))
            scores.append(final_score)
            
            logger.info(f"Document {i+1}: base={base_score}, boost={content_boost}, final={final_score}")
        
        logger.info(f"Scores de fallback générés: {scores}")
        return scores

    def _intelligent_rerank_without_llm(self, results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Réordonnancement intelligent sans LLM, spécialement optimisé pour les requêtes bancaires.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            limit: Nombre maximum de résultats à retourner
            
        Returns:
            Liste réordonnée des résultats
        """
        if not results:
            return []
        
        logger.info(f"Réordonnancement intelligent sans LLM pour la requête: {query}")
        
        query_lower = query.lower()
        
        # Analyser la requête pour extraire les mots-clés pertinents
        query_words = query_lower.split()
        
        # Mots-clés pour différents types de documents (plus génériques)
        document_type_keywords = {
            'bancaire': ['rib', 'relevé', 'identité', 'bancaire', 'iban', 'bic', 'banque', 'compte', 'coordonnées'],
            'facture': ['facture', 'note', 'devis', 'avoir', 'facturation', 'montant', 'euros', '€', 'prix', 'total'],
            'sécurité_sociale': ['sécurité', 'sociale', 'securite', 'caf', 'cpam', 'assurance', 'maladie', 'retraite', 'pension', 'social'],
            'général': query_words  # Utiliser les mots de la requête comme mots-clés généraux
        }
        
        # Déterminer le type de document recherché de manière plus flexible
        detected_types = []
        for doc_type, keywords in document_type_keywords.items():
            if any(word in query_lower for word in keywords):
                detected_types.append(doc_type)
        
        reranked_results = []
        
        for i, result in enumerate(results):
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "").lower()
            filename = payload.get("filename", "").lower()
            original_score = result.get("score", 0)
            
            # Score de base
            base_score = original_score
            
            # Boost basé sur le contenu du document
            content_boost = 0
            
            # Calculer le boost basé sur les types détectés
            for doc_type in detected_types:
                if doc_type == 'général':
                    # Pour les mots-clés généraux, boost modéré
                    matches = sum(1 for word in query_words if word in text or word in filename)
                    content_boost += min(1.0, matches * 0.3)
                else:
                    # Pour les types spécifiques, boost plus fort
                    keywords = document_type_keywords[doc_type]
                    matches = sum(1 for word in keywords if word in text or word in filename)
                    if matches > 0:
                        content_boost += min(2.0, matches * 0.5)  # Boost jusqu'à 2 points
                        logger.info(f"Document {i+1} ({filename}): {matches} mots-clés {doc_type} trouvés, boost: +{min(2.0, matches * 0.5)}")
            
            # Si aucun type spécifique détecté, utiliser une recherche générale
            if not detected_types or detected_types == ['général']:
                matches = sum(1 for word in query_words if word in text or word in filename)
                content_boost = min(1.0, matches * 0.3)
            
            # Score final
            final_score = max(0.1, base_score + content_boost)
            
            # Créer une copie du résultat avec le nouveau score
            new_result = result.copy()
            new_result["score"] = final_score
            new_result["intelligent_boost"] = {
                "base_score": base_score,
                "content_boost": content_boost,
                "final_score": final_score
            }
            
            reranked_results.append(new_result)
        
        # Réordonner selon le nouveau score
        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        logger.info(f"Réordonnancement intelligent terminé: {len(reranked_results)} résultats triés")
        
        return reranked_results[:limit]

    def _rerank_with_llm(self, results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Réordonne les résultats en utilisant un LLM pour évaluer leur pertinence.
        Version optimisée qui traite plusieurs documents en une seule fois.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            limit: Nombre maximum de résultats à retourner
            
        Returns:
            Liste réordonnée des résultats
        """
        if not results:
            return []
        
        logger.info(f"Réordonnancement LLM optimisé avec {len(results)} résultats pour la requête: {query}")
        
        # Préparer les documents avec leurs textes
        documents_with_text = []
        for i, result in enumerate(results):
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
            
            # Tronquer le texte pour l'envoi au LLM
            max_length = config_service.search.max_document_length
            if len(document_text) > max_length:
                document_text = document_text[:max_length] + "..."
            
            if document_text:
                documents_with_text.append({
                    "index": i,
                    "result": result,
                    "text": document_text,
                    "id": result.get("id", f"doc_{i}")
                })
            else:
                logger.warning(f"Document {i+1}/{len(results)} sans texte exploitable - conservation du score original")
        
        if not documents_with_text:
            logger.warning("Aucun document avec texte trouvé pour le réordonnancement")
            return results[:limit]
        
        # Construire le prompt optimisé pour traiter plusieurs documents
        documents_text = ""
        for i, doc in enumerate(documents_with_text):
            # Limiter la taille de chaque document pour éviter les dépassements
            doc_text = doc['text'][:500]  # Limiter à 500 caractères par document
            documents_text += f"\n--- Document {i+1} (ID: {doc['id']}) ---\n{doc_text}\n"
        
        # Prompt ultra-simplifié pour éviter les problèmes de parsing
        batch_prompt = f"""Score chaque document de 0 à 10 pour: "{query}"

{documents_text}

Réponds seulement avec des nombres séparés par des virgules.
Exemple: 8,6,4,9,7

Scores:"""

        try:
            # Appel au LLM via l'API de chat avec un prompt plus simple
            response = requests.post(
                f"{config_service.llm.base_url}/api/chat",
                json={
                    "model": config_service.llm.model,
                    "messages": [
                        {"role": "system", "content": "Tu es un expert en évaluation de documents. Tu dois répondre UNIQUEMENT avec des nombres de 0 à 10 séparés par des virgules. Aucun autre texte."},
                        {"role": "user", "content": batch_prompt}
                    ],
                    "temperature": 0.0,  # Température 0 pour plus de cohérence
                    "max_tokens": 100  # Limiter la réponse
                },
                timeout=30
            )
            
            # Parser la réponse
            try:
                # Le LLM peut retourner plusieurs objets JSON séparés par des sauts de ligne
                raw_response = response.text.strip()
                
                # Si la réponse contient plusieurs objets JSON, prendre le premier
                if '\n' in raw_response:
                    first_json = raw_response.split('\n')[0]
                    response_json = json.loads(first_json)
                else:
                    response_json = response.json()
                
                llm_response = response_json.get("message", {}).get("content", "")
                
                # Si pas de contenu dans message, essayer d'autres champs
                if not llm_response:
                    llm_response = response_json.get("response", "")
                if not llm_response:
                    llm_response = response_json.get("content", "")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Erreur de parsing JSON de la réponse LLM: {e}")
                logger.error(f"Réponse brute: {response.text[:200]}...")
                return results[:limit]
            
            # Extraire les scores
            scores_text = llm_response.strip()
            logger.info(f"Réponse LLM brute: {scores_text}")
            
            # Vérifier si la réponse contient du texte au lieu de scores
            if (len(scores_text) > 50 or 
                any(word in scores_text.lower() for word in ['analyser', 'document', 'pertinent', 'recherche', 'trouver', 'désolé', 'après', 'avoir', 'trouvé', 'aucun', 'renseignement']) or
                scores_text == "0" or
                not any(char.isdigit() for char in scores_text)):
                logger.warning("Le LLM a retourné du texte au lieu de scores, utilisation du fallback intelligent")
                scores = self._generate_intelligent_fallback_scores(documents_with_text, query)
            else:
                # Parser les scores (supporter différents formats)
                scores = []
                
                # Nettoyer la réponse - enlever tout texte non numérique
                cleaned_text = re.sub(r'[^\d,.\s]', '', scores_text)
                
                if ',' in cleaned_text:
                    # Format: "8,6,4,9,7"
                    score_parts = cleaned_text.split(',')
                    for part in score_parts:
                        digits = re.findall(r'\d+', part.strip())
                        if digits:
                            score = int(digits[0])
                            scores.append(max(0, min(10, score)))  # Clamp entre 0 et 10
                else:
                    # Format: "8 6 4 9 7" ou autres
                    digits = re.findall(r'\d+', cleaned_text)
                    scores = [max(0, min(10, int(d))) for d in digits]
                
                # Si toujours pas de scores valides, essayer de récupérer n'importe quel nombre
                if not scores:
                    all_numbers = re.findall(r'\d+', scores_text)
                    if all_numbers:
                        scores = [max(0, min(10, int(num))) for num in all_numbers]
                        logger.warning(f"Récupération de scores de secours: {scores}")
                
                # Si toujours pas de scores, utiliser des scores basés sur la position
                if not scores:
                    logger.warning("Aucun score valide trouvé, utilisation de scores par défaut basés sur la position")
                    scores = [max(3, 8 - i) for i in range(len(documents_with_text))]
            
            # S'assurer qu'on a le bon nombre de scores
            if len(scores) != len(documents_with_text):
                logger.warning(f"Nombre de scores ({len(scores)}) ne correspond pas au nombre de documents ({len(documents_with_text)})")
                
                # Si on a trop de scores, prendre les premiers
                if len(scores) > len(documents_with_text):
                    scores = scores[:len(documents_with_text)]
                # Si on n'a pas assez de scores, compléter avec des scores par défaut
                else:
                    while len(scores) < len(documents_with_text):
                        # Utiliser un score basé sur la position (les premiers documents sont généralement plus pertinents)
                        default_score = max(3, 8 - len(scores))  # Score décroissant de 8 à 3
                        scores.append(default_score)
                
                logger.info(f"Scores ajustés: {scores}")
            
            # Appliquer les scores aux documents
            reranked_results = []
            for i, doc in enumerate(documents_with_text):
                original_score = doc["result"].get("score", 0)
                llm_score = scores[i] if i < len(scores) else 5
                
                # Score combiné avec les poids configurés
                normalized_original = min(10, original_score * 10)
                combined_score = (llm_score * config_service.search.llm_score_weight) + (normalized_original * config_service.search.original_score_weight)
                
                # Créer une copie du résultat avec les nouveaux scores
                new_result = doc["result"].copy()
                new_result["llm_score"] = llm_score
                new_result["score"] = combined_score
                
                reranked_results.append(new_result)
            
            # Ajouter les documents sans texte avec leur score original
            for i, result in enumerate(results):
                if not any(doc["index"] == i for doc in documents_with_text):
                    reranked_results.append(result)
            
            # Réordonner selon le score combiné
            reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Réordonnancement terminé: {len(reranked_results)} résultats triés")
            
            return reranked_results[:limit]
            
        except Exception as e:
            logger.error(f"Erreur lors du réordonnancement LLM optimisé: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # En cas d'erreur, retourner les résultats originaux triés par score
            logger.warning("Fallback vers tri par score original sans LLM")
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
            return results[:limit]

    def _generate_answer(self, results: List[Dict[str, Any]], query: str) -> str:
        """
        Génère une réponse basée sur le contexte des documents trouvés.
        Utilise directement le LLM avec des param��tres optimisés pour llama3.1.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            
        Returns:
            Réponse générée
        """
        if not results:
            return "Je n'ai pas trouvé d'information pertinente pour répondre à votre question."
        
        # 1. Extraction et optimisation du contexte des documents
        context_parts = []
        has_content = False
        
        # Utiliser les meilleurs résultats (jusqu'à 5 pour plus de contexte)
        top_results = results[:5]
        
        for i, result in enumerate(top_results):
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
            
            # Formater le document pour le contexte avec optimisation
            if doc_text:
                has_content = True
                
                # Optimiser la taille du texte selon le score
                score = result.get("score", 0)
                if score > 0.8:  # Documents très pertinents
                    max_length = config_service.search.max_document_length
                elif score > 0.6:  # Documents moyennement pertinents
                    max_length = int(config_service.search.max_document_length * 0.8)
                else:  # Documents moins pertinents
                    max_length = int(config_service.search.max_document_length * 0.6)
                
                if len(doc_text) > max_length:
                    # Tronquer intelligemment (essayer de couper à la fin d'une phrase)
                    truncated = doc_text[:max_length]
                    last_period = truncated.rfind('.')
                    last_exclamation = truncated.rfind('!')
                    last_question = truncated.rfind('?')
                    
                    # Prendre la dernière ponctuation trouvée
                    last_sentence_end = max(last_period, last_exclamation, last_question)
                    if last_sentence_end > max_length * 0.7:  # Si on ne perd pas trop de contenu
                        doc_text = truncated[:last_sentence_end + 1]
                    else:
                        doc_text = truncated + "..."
                
                # Ajouter des métadonnées utiles
                metadata_info = []
                if score > 0:
                    metadata_info.append(f"Score de pertinence: {score:.2f}")
                
                # Ajouter des informations sur les entités si disponibles
                if "llm_entities" in payload and payload["llm_entities"]:
                    entities = payload["llm_entities"][:3]  # Limiter à 3 entités
                    entity_texts = [ent.get("text", "") for ent in entities if isinstance(ent, dict)]
                    if entity_texts:
                        metadata_info.append(f"Entités clés: {', '.join(entity_texts)}")
                
                metadata_str = f" ({', '.join(metadata_info)})" if metadata_info else ""
                
                context_parts.append(f"Document {i+1}: {title}{metadata_str}\n{doc_text}")
                logger.info(f"Document {i+1} ajouté au contexte ({len(doc_text)} caractères, score: {score:.2f})")
            else:
                logger.warning(f"Document {i+1} ignoré (pas de texte)")
        
        # 2. Vérifier si on a du contenu exploitable
        if not has_content:
            return "J'ai trouvé des documents qui pourraient correspondre à votre recherche, mais je n'ai pas pu extraire leur contenu textuel. Veuillez vérifier le format des documents."
        
        # 3. Assembler le contexte complet
        context_text = "\n\n".join(context_parts)
        
        # 4. Préparer l'appel au LLM
        if "llama" in config_service.llm.model.lower():
            # Construction des messages pour l'API Chat avec prompt optimisé
            messages = [
                {"role": "system", "content": """Tu es un assistant IA expert en analyse de documents administratifs français. Ta mission est de :

1. ANALYSER précisément le contenu des documents fournis
2. RÉPONDRE uniquement en te basant sur les informations présentes dans ces documents
3. NE PAS faire d'hypothèses ou de suppositions non vérifiées
4. SI une information n'est pas claire ou manquante, le dire explicitement
5. CITER les documents sources pour chaque information importante
6. NE PAS confondre les documents entre eux
7. ÊTRE particulièrement attentif aux détails et aux nuances
8. PRIVILÉGIER la précision à la quantité d'informations

FORMAT DE RÉPONSE ATTENDU :
- Réponse directe et structurée à la question
- Citations précises des documents sources (ex: "Selon le Document 1...")
- Précision si une information est manquante ou ambiguë
- Distinction claire entre les faits et les interprétations
- Utilisation d'un langage clair et professionnel

IMPORTANT : Base-toi UNIQUEMENT sur les documents fournis. Ne pas ajouter d'informations externes."""},
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

    def search_document_by_name(
        self,
        document_name: str,
        limit: int = 5,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Recherche un document spécifique par son nom (recherche exacte ou partielle).
        Délègue à la méthode simplifiée pour éviter les problèmes de réordonnancement LLM.
        """
        return self.search_document_by_name_simple(document_name, limit, collection_name)

    def search_document_by_name_simple(
        self,
        document_name: str,
        limit: int = 5,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Recherche simplifiée d'un document par son nom (sans réordonnancement LLM).
        Utilise une recherche directe dans Qdrant avec filtrage par nom de fichier.
        
        Args:
            document_name: Nom du document à rechercher
            limit: Nombre maximum de résultats
            collection_name: Nom de la collection (par défaut self.collection_name)
            
        Returns:
            Dictionnaire contenant les résultats de recherche
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            # Récupérer TOUS les documents de la collection pour filtrer par nom
            logger.info(f"Recherche de documents avec le nom '{document_name}' dans la collection '{collection_name}'")
            
            # Utiliser scroll pour récupérer tous les documents
            response = self.qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=1000,  # Récupérer un grand nombre de documents
                with_payload=True,
                with_vectors=False  # Pas besoin des vecteurs pour le filtrage par nom
            )
            
            # Extraire les points de la réponse
            points = response[0]
            logger.info(f"Récupéré {len(points)} documents de la collection")
            
            if not points:
                return {"results": [], "total_found": 0, "query": document_name}
            
            # Filtrer par nom de fichier
            filtered_results = []
            document_name_lower = document_name.lower()
            
            for point in points:
                payload = point.payload if point.payload else {}
                filename = payload.get("filename", "").lower()
                
                # Recherche partielle dans le nom de fichier
                if document_name_lower in filename:
                    # Créer un résultat au format attendu
                    result = {
                        "id": point.id,
                        "score": 1.0,  # Score par défaut pour les correspondances exactes
                        "payload": payload
                    }
                    
                    # Boost le score pour les correspondances exactes
                    if filename == document_name_lower:
                        result["score"] = 1.5
                    elif filename.startswith(document_name_lower):
                        result["score"] = 1.2
                    
                    filtered_results.append(result)
                    logger.info(f"Document trouvé: {filename} (score: {result['score']})")
            
            # Trier par score décroissant
            filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            logger.info(f"Trouvé {len(filtered_results)} documents correspondant à '{document_name}'")
            
            return {
                "results": filtered_results[:limit],
                "total_found": len(filtered_results),
                "query": document_name
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche simple par nom de document: {str(e)}")
            return {
                "error": str(e),
                "results": [],
                "total_found": 0,
                "query": document_name
            }

    def simple_vector_search(
        self,
        query: str,
        limit: int = 10,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Recherche vectorielle simple sans réordonnancement LLM.
        
        Args:
            query: Requête de recherche
            limit: Nombre maximum de résultats
            collection_name: Nom de la collection
            
        Returns:
            Dictionnaire contenant les résultats de recherche
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            # Recherche vectorielle directe
            query_vector = embedding_service.get_embedding(query)
            if query_vector is None:
                logger.error("Impossible de générer un embedding pour la requête")
                return {"results": [], "total_found": 0, "query": query}

            # Recherche dans Qdrant
            search_results = self.qdrant_service.search_similar(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )

            if not search_results:
                return {"results": [], "total_found": 0, "query": query}
            
            return {
                "results": search_results,
                "total_found": len(search_results),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche vectorielle simple: {str(e)}")
            return {
                "error": str(e),
                "results": [],
                "total_found": 0,
                "query": query
            }

    def search_document_by_name_efficient(
        self,
        document_name: str,
        limit: int = 5,
        collection_name: str = None
    ) -> Dict[str, Any]:
        """
        Recherche efficace d'un document par son nom en utilisant une approche hybride.
        Essaie d'abord une recherche vectorielle, puis une recherche par scroll si nécessaire.
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            logger.info(f"Recherche efficace de documents avec le nom '{document_name}'")
            
            # Étape 1: Essayer une recherche vectorielle avec le nom du document
            query_vector = embedding_service.get_embedding(document_name)
            if query_vector is not None:
                search_results = self.qdrant_service.search_similar(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit * 2
                )
                
                if search_results:
                    # Filtrer les résultats par nom de fichier
                    filtered_results = []
                    document_name_lower = document_name.lower()
                    
                    for result in search_results:
                        payload = result.get("payload", {})
                        filename = payload.get("filename", "").lower()
                        
                        if document_name_lower in filename:
                            # Boost le score pour les correspondances exactes
                            if filename == document_name_lower:
                                result["score"] = result.get("score", 0) * 1.5
                            elif filename.startswith(document_name_lower):
                                result["score"] = result.get("score", 0) * 1.2
                            
                            filtered_results.append(result)
                    
                    if filtered_results:
                        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                        logger.info(f"Trouvé {len(filtered_results)} documents via recherche vectorielle")
                        return {
                            "results": filtered_results[:limit],
                            "total_found": len(filtered_results),
                            "query": document_name
                        }
            
            # Étape 2: Si pas de résultats, utiliser la méthode par scroll
            logger.info("Recherche vectorielle insuffisante, utilisation de la méthode par scroll")
            return self.search_document_by_name_simple(document_name, limit, collection_name)
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche efficace par nom de document: {str(e)}")
            return {
                "error": str(e),
                "results": [],
                "total_found": 0,
                "query": document_name
            }

    def generate_document_response(
        self,
        document: Dict[str, Any],
        question: str,
        discussion_id: Optional[str] = None,
        settings_id: Optional[str] = None,
        include_context: bool = True,
        max_context_length: int = 5000
    ) -> Dict[str, Any]:
        """
        Génère une réponse contextuelle basée sur un document spécifique.
        
        Args:
            document: Document trouvé par la recherche
            question: Question à poser sur le document
            discussion_id: ID de la discussion pour le contexte
            settings_id: ID des paramètres à utiliser
            include_context: Inclure le contexte de la discussion
            max_context_length: Longueur maximale du contexte
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        try:
            # 1. Extraire le contenu du document
            document_text = self._extract_document_text(document)
            if not document_text:
                return {
                    "answer": "Je n'ai pas pu extraire le contenu de ce document pour répondre à votre question.",
                    "context_used": False,
                    "discussion_context": False
                }
            
            # 2. Construire le contexte de base
            context_parts = []
            
            # Ajouter le contenu du document
            document_title = document.get("payload", {}).get("filename", "Document")
            context_parts.append(f"Document: {document_title}\n{document_text}")
            
            # 3. Ajouter le contexte de la discussion si demandé
            discussion_context = ""
            if include_context and discussion_id:
                try:
                    from .discussions_service import discussions_service
                    discussion_context = discussions_service.get_discussion_context(
                        discussion_id=discussion_id,
                        max_length=max_context_length // 2  # Utiliser la moitié pour la discussion
                    )
                    if discussion_context:
                        context_parts.append(f"Contexte de la discussion:\n{discussion_context}")
                except Exception as e:
                    logger.warning(f"Impossible de récupérer le contexte de la discussion: {e}")
            
            # 4. Assembler le contexte complet
            full_context = "\n\n".join(context_parts)
            
            # 5. Tronquer si nécessaire
            if len(full_context) > max_context_length:
                full_context = full_context[:max_context_length] + "..."
            
            # 6. Générer la réponse avec le LLM
            answer = self._generate_contextual_answer(
                context=full_context,
                question=question,
                document_title=document_title
            )
            
            return {
                "answer": answer,
                "context_used": True,
                "discussion_context": bool(discussion_context),
                "document_length": len(document_text),
                "total_context_length": len(full_context)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse documentaire: {str(e)}")
            return {
                "answer": f"Une erreur est survenue lors de l'analyse du document: {str(e)}",
                "context_used": False,
                "discussion_context": False
            }

    def _extract_document_text(self, document: Dict[str, Any]) -> str:
        """
        Extrait le texte d'un document de manière robuste.
        
        Args:
            document: Document à analyser
            
        Returns:
            Texte extrait du document
        """
        # Vérifier d'abord si le document a directement un champ "cleaned_text"
        if "cleaned_text" in document and isinstance(document["cleaned_text"], str) and document["cleaned_text"].strip():
            return document["cleaned_text"].strip()
        
        # Chercher dans le payload
        payload = document.get("payload", {})
        if isinstance(payload, dict):
            # Champs directs possibles
            text_fields = ["cleaned_text", "text", "content", "body", "full_text"]
            for field in text_fields:
                if field in payload and isinstance(payload[field], str) and payload[field].strip():
                    return payload[field].strip()
            
            # Chercher dans les métadonnées
            if "metadata" in payload and isinstance(payload["metadata"], dict):
                metadata = payload["metadata"]
                for field in text_fields:
                    if field in metadata and isinstance(metadata[field], str) and metadata[field].strip():
                        return metadata[field].strip()
        
        return ""

    def _generate_contextual_answer(
        self,
        context: str,
        question: str,
        document_title: str
    ) -> str:
        """
        Génère une réponse contextuelle en utilisant le LLM.
        
        Args:
            context: Contexte complet (document + discussion)
            question: Question à poser
            document_title: Titre du document
            
        Returns:
            Réponse générée par le LLM
        """
        try:
            if "llama" in config_service.llm.model.lower():
                # Construction des messages pour l'API Chat
                messages = [
                    {"role": "system", "content": f"""Tu es un assistant spécialisé dans l'analyse de documents. Tu as accès au document "{document_title}" et tu peux répondre à des questions précises sur son contenu.

Instructions importantes :
1. Réponds UNIQUEMENT en te basant sur le contenu du document fourni
2. Si une information n'est pas dans le document, dis-le clairement
3. Cite des passages spécifiques du document quand c'est pertinent
4. Sois précis et factuel
5. Si le contexte de discussion est fourni, prends-le en compte pour mieux comprendre la question
6. Structure ta réponse de manière claire et organisée

Format de réponse :
- Réponse directe à la question
- Citations du document quand approprié
- Précision si une information n'est pas disponible dans le document"""},
                    {"role": "user", "content": f"""Contexte du document et de la discussion :

{context}

Question : {question}

Analyse le document et réponds à la question en te basant uniquement sur son contenu. Si le contexte de discussion aide à comprendre la question, utilise-le également."""}
                ]
                
                # Appel à l'API de chat
                response = requests.post(
                    f"{config_service.llm.base_url}/api/chat",
                    json={
                        "model": config_service.llm.model,
                        "messages": messages,
                        "temperature": config_service.llm.temperature,
                        "stream": False
                    },
                    timeout=60
                )
                
                # Parser la réponse
                try:
                    response_json = response.json()
                    answer = response_json.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    logger.error("Erreur de parsing JSON de la réponse LLM")
                    answer = "Je n'ai pas pu générer une réponse à cause d'un problème technique."
                
            else:
                # Pour d'autres modèles, utiliser l'API de génération
                prompt = f"""Contexte du document et de la discussion :

{context}

Question : {question}

Analyse le document et réponds à la question en te basant uniquement sur son contenu :"""
                
                response = requests.post(
                    f"{config_service.llm.base_url}/api/generate",
                    json={
                        "model": config_service.llm.model,
                        "prompt": prompt,
                        "temperature": config_service.llm.temperature,
                        "stream": False
                    },
                    timeout=60
                )
                
                try:
                    response_json = response.json()
                    answer = response_json.get("response", "")
                except json.JSONDecodeError:
                    logger.error("Erreur de parsing JSON de la réponse LLM")
                    answer = "Je n'ai pas pu générer une réponse à cause d'un problème technique."
            
            return answer if answer else "Je n'ai pas pu générer une réponse basée sur le document."
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse contextuelle: {str(e)}")
            return f"Une erreur est survenue lors de la génération de la réponse: {str(e)}"


# Instance singleton du service
search_service = SearchService()
