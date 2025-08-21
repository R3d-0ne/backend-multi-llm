"""
Classes spécialisées pour les fonctionnalités de recherche.
Découpe la logique de recherche en composants plus petits et plus facilement testables.
"""
import re
import requests
import json
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .base_service import BaseService, LogLevel, ServiceResponse, ExternalServiceError
from .config_service import config_service
from ..libs.functions.global_functions import (
    tokenize_text, 
    remove_stopwords,
    create_sparse_vector,
    extract_entities_advanced
)


@dataclass
class SearchResult:
    """Représentation structurée d'un résultat de recherche"""
    id: str
    score: float
    payload: Dict[str, Any]
    llm_score: Optional[float] = None
    keyword_boost: Optional[Dict[str, Any]] = None
    

class DocumentProcessor(BaseService):
    """
    Service pour le traitement et l'extraction de contenu des documents.
    """
    
    def __init__(self):
        super().__init__("DocumentProcessor")
    
    def extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """
        Extrait le texte d'un résultat de recherche.
        
        Args:
            result: Résultat de recherche
            
        Returns:
            Texte extrait ou chaîne vide
        """
        # Vérifier d'abord si le document a directement un champ "cleaned_text"
        if "cleaned_text" in result and isinstance(result["cleaned_text"], str) and result["cleaned_text"].strip():
            return result["cleaned_text"].strip()
        
        # Chercher dans le payload
        payload = result.get("payload", {})
        if isinstance(payload, dict):
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
    
    def extract_title_from_result(self, result: Dict[str, Any], index: int = 0) -> str:
        """
        Extrait le titre d'un résultat de recherche.
        
        Args:
            result: Résultat de recherche
            index: Index du document (pour titre par défaut)
            
        Returns:
            Titre du document
        """
        payload = result.get("payload", {})
        
        # Essayer différents champs pour le titre
        title_fields = ["filename", "title", "name"]
        for field in title_fields:
            if field in payload and isinstance(payload[field], str) and payload[field].strip():
                return payload[field].strip()
        
        # Chercher dans les métadonnées
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            metadata = payload["metadata"]
            for field in title_fields:
                if field in metadata and isinstance(metadata[field], str) and metadata[field].strip():
                    return metadata[field].strip()
        
        # Titre par défaut
        score = result.get("score", 0)
        return f"Document {index + 1} (score: {score:.2f})"
    
    def health_check(self) -> ServiceResponse:
        """Vérifie l'état de santé du processeur de documents"""
        return ServiceResponse.success_response({"status": "healthy", "service": "DocumentProcessor"})


class KeywordBooster(BaseService):
    """
    Service pour augmenter le score des résultats contenant des mots-clés de la requête.
    """
    
    def __init__(self):
        super().__init__("KeywordBooster")
        self.config = config_service.search
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extrait les mots-clés significatifs d'une requête.
        
        Args:
            query: Requête de recherche
            
        Returns:
            Liste des mots-clés
        """
        # Extraction des mots-clés significatifs (plus de 3 caractères)
        keywords = [word.lower() for word in re.findall(r'\b\w{3,}\b', query)]
        
        # Extraction des entités nommées de la requête
        try:
            entities_result = extract_entities_advanced(query)
            entities = [ent['text'].lower() for ent in entities_result.get('entities', [])]
        except Exception as e:
            self.log(LogLevel.WARNING, f"Erreur lors de l'extraction des entités: {e}")
            entities = []
        
        # Combiner les mots-clés et les entités (éliminer les doublons)
        all_keywords = list(set(keywords + entities))
        
        self.log(LogLevel.DEBUG, f"Mots-clés extraits: {all_keywords}")
        return all_keywords
    
    def calculate_keyword_scores(self, text: str, keywords: List[str]) -> Dict[str, float]:
        """
        Calcule différents scores basés sur les mots-clés.
        
        Args:
            text: Texte à analyser
            keywords: Liste des mots-clés
            
        Returns:
            Dictionnaire avec les différents scores
        """
        if not keywords:
            return {"keyword_score": 0.0, "proximity_score": 0.0, "density_score": 0.0}
        
        text_lower = text.lower()
        
        # Score de présence des mots-clés
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_score = keyword_matches / len(keywords)
        
        # Score de proximité (position des mots-clés)
        proximity_score = 0.0
        if keyword_matches > 0:
            positions = []
            for keyword in keywords:
                if keyword in text_lower:
                    pos = text_lower.find(keyword)
                    positions.append(pos)
            
            if positions:
                avg_position = sum(positions) / len(positions)
                # Plus les mots-clés sont proches du début, plus le score est élevé
                proximity_score = 1 - (avg_position / len(text_lower)) if len(text_lower) > 0 else 0
        
        # Score de densité des mots-clés
        density_score = 0.0
        if keyword_matches > 0 and len(text_lower) > 0:
            # Densité des mots-clés (nombre de mots-clés par 100 caractères)
            density = (keyword_matches * 100) / len(text_lower)
            density_score = min(1.0, density / 5.0)  # Normaliser sur une densité maximale de 5%
        
        return {
            "keyword_score": keyword_score,
            "proximity_score": proximity_score,
            "density_score": density_score
        }
    
    def boost_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Augmente le score des résultats basé sur la présence de mots-clés.
        
        Args:
            results: Liste des résultats de recherche
            query: Requête de recherche
            
        Returns:
            Liste des résultats avec scores ajustés
        """
        keywords = self.extract_keywords(query)
        if not keywords:
            self.log(LogLevel.DEBUG, "Aucun mot-clé trouvé, pas de boost appliqué")
            return results
        
        for result in results:
            # Extraire le texte du document
            payload = result.get("payload", {})
            text = payload.get("cleaned_text", "")
            
            if not text:
                continue
            
            # Calculer les scores de mots-clés
            scores = self.calculate_keyword_scores(text, keywords)
            
            # Combiner les scores avec pondération
            final_boost = (
                scores["keyword_score"] * 0.5 +      # 50% pour la présence
                scores["proximity_score"] * 0.3 +     # 30% pour la proximité
                scores["density_score"] * 0.2         # 20% pour la densité
            )
            
            # Application du boost (maximum configurable)
            boost = 1 + (final_boost * self.config.keyword_boost_max)
            original_score = result.get("score", 0)
            result["score"] = original_score * boost
            
            # Ajouter l'information sur le boost pour le débogage
            result["keyword_boost"] = {
                "boost": boost,
                **scores
            }
        
        # Réordonner les résultats selon le nouveau score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        self.log(LogLevel.DEBUG, f"Boost de mots-clés appliqué à {len(results)} résultats")
        return results
    
    def health_check(self) -> ServiceResponse:
        """Vérifie l'état de santé du booster de mots-clés"""
        return ServiceResponse.success_response({"status": "healthy", "service": "KeywordBooster"})


class LLMReranker(BaseService):
    """
    Service pour réordonner les résultats en utilisant un LLM.
    """
    
    def __init__(self):
        super().__init__("LLMReranker")
        self.config = config_service.llm
        self.search_config = config_service.search
        self.document_processor = DocumentProcessor()
    
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Appelle l'API LLM pour obtenir une réponse.
        
        Args:
            messages: Messages à envoyer au LLM
            
        Returns:
            Réponse du LLM ou None en cas d'erreur
        """
        try:
            api_url = f"{self.config.base_url}/api/chat"
            payload = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature
            }
            
            self.log(LogLevel.DEBUG, f"Appel LLM API: {api_url}")
            
            response = requests.post(
                api_url,
                json=payload,
                timeout=self.config.timeout
            )
            
            if not response.ok:
                raise ExternalServiceError(
                    f"Erreur API LLM: {response.status_code}",
                    "LLM",
                    response.status_code
                )
            
            # Traitement de la réponse
            try:
                response_json = response.json()
                return response_json.get("message", {}).get("content", "")
            except json.JSONDecodeError:
                # Essayer d'extraire la réponse manuellement
                raw_response = response.text
                if '"content": "' in raw_response:
                    content_start = raw_response.find('"content": "') + 12
                    content_end = raw_response.find('"', content_start)
                    if content_end > content_start:
                        return raw_response[content_start:content_end]
                
                self.log(LogLevel.WARNING, "Impossible de parser la réponse JSON du LLM")
                return None
                
        except requests.Timeout:
            raise ExternalServiceError("Timeout lors de l'appel au LLM", "LLM")
        except requests.RequestException as e:
            raise ExternalServiceError(f"Erreur de connexion au LLM: {str(e)}", "LLM")
    
    def _extract_score_from_response(self, llm_response: str) -> Optional[float]:
        """
        Extrait le score numérique de la réponse du LLM.
        
        Args:
            llm_response: Réponse du LLM
            
        Returns:
            Score numérique ou None si extraction impossible
        """
        try:
            # Essayer d'extraire uniquement les chiffres
            digits = re.findall(r'\d+', llm_response)
            if digits:
                score = int(digits[0])
            else:
                score = int(llm_response.strip())
            
            # Normaliser le score dans les limites 0-10
            return max(0, min(10, score))
            
        except (ValueError, AttributeError):
            self.log(LogLevel.WARNING, f"Impossible d'extraire le score de: '{llm_response}'")
            return None
    
    def _calculate_combined_score(self, llm_score: float, original_score: float, result: Dict[str, Any]) -> float:
        """
        Calcule le score combiné entre LLM et score original.
        
        Args:
            llm_score: Score donné par le LLM (0-10)
            original_score: Score original de la recherche vectorielle
            result: Résultat complet (pour ajustements additionnels)
            
        Returns:
            Score combiné
        """
        # Normaliser le score original à 10
        normalized_original = min(10, original_score * 10)
        
        # Score combiné avec pondération configurable
        combined_score = (
            llm_score * self.search_config.llm_score_weight + 
            normalized_original * self.search_config.original_score_weight
        )
        
        # Ajustement basé sur la proximité des mots-clés si disponible
        if "keyword_boost" in result:
            keyword_boost = result["keyword_boost"]
            if isinstance(keyword_boost, dict):
                proximity_factor = keyword_boost.get("proximity_score", 0)
                combined_score *= (1 + (proximity_factor * self.search_config.proximity_adjustment_max))
        
        return combined_score
    
    def rerank_results(self, results: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Réordonne les résultats en utilisant le LLM.
        
        Args:
            results: Liste des résultats à réordonner
            query: Requête de recherche
            limit: Nombre maximum de résultats à retourner
            
        Returns:
            Liste réordonnée des résultats
        """
        if not results:
            return []
        
        reranked_results = []
        
        self.log(LogLevel.INFO, f"Réordonnancement LLM de {len(results)} résultats pour: {query}")
        
        for i, result in enumerate(results):
            try:
                # Extraire le texte du document
                document_text = self.document_processor.extract_text_from_result(result)
                
                if not document_text:
                    self.log(LogLevel.WARNING, f"Document {i+1} sans texte - conservation du score original")
                    reranked_results.append(result)
                    continue
                
                # Tronquer le texte si nécessaire
                truncated_text = self.document_processor.truncate_text(
                    document_text, 
                    self.search_config.max_document_length
                )
                
                # Construire le prompt pour le LLM
                prompt = f"Requête: {query}\n\nDocument: {truncated_text}\n\nScore de pertinence (0-10):"
                
                messages = [
                    {"role": "system", "content": self.config.reranker_system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                # Appeler le LLM
                llm_response = self._call_llm_api(messages)
                
                if llm_response is None:
                    self.log(LogLevel.WARNING, f"Pas de réponse LLM pour le document {i+1}")
                    reranked_results.append(result)
                    continue
                
                # Extraire le score
                llm_score = self._extract_score_from_response(llm_response)
                
                if llm_score is None:
                    self.log(LogLevel.WARNING, f"Score LLM invalide pour le document {i+1}: '{llm_response}'")
                    reranked_results.append(result)
                    continue
                
                # Calculer le score combiné
                original_score = result.get("score", 0)
                combined_score = self._calculate_combined_score(llm_score, original_score, result)
                
                # Créer le nouveau résultat
                new_result = result.copy()
                new_result["llm_score"] = llm_score
                new_result["score"] = combined_score
                
                reranked_results.append(new_result)
                
                self.log(LogLevel.DEBUG, f"Document {i+1} - Score LLM: {llm_score}, Score combiné: {combined_score:.2f}")
                
            except Exception as e:
                self.log(LogLevel.ERROR, f"Erreur lors du réordonnancement du document {i+1}: {str(e)}")
                reranked_results.append(result)
        
        # Réordonner selon le score combiné
        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        self.log(LogLevel.INFO, f"Réordonnancement terminé: {len(reranked_results)} résultats triés")
        
        # Limiter au nombre demandé
        return reranked_results[:limit]
    
    def health_check(self) -> ServiceResponse:
        """Vérifie l'état de santé du réordonnanceur LLM"""
        try:
            # Test simple d'appel au LLM
            test_messages = [
                {"role": "system", "content": "Réponds simplement 'OK'"},
                {"role": "user", "content": "Test"}
            ]
            
            response = self._call_llm_api(test_messages)
            
            if response is not None:
                return ServiceResponse.success_response({
                    "status": "healthy", 
                    "service": "LLMReranker",
                    "llm_accessible": True
                })
            else:
                return ServiceResponse.error_response(
                    "LLM non accessible",
                    "LLM_UNAVAILABLE"
                )
                
        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur lors du test LLM: {str(e)}",
                "LLM_ERROR"
            )


class SearchResultFilter(BaseService):
    """
    Service pour filtrer les résultats de recherche selon des critères.
    """
    
    def __init__(self):
        super().__init__("SearchResultFilter")
    
    def apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Applique des filtres sur les résultats de recherche.
        
        Args:
            results: Liste des résultats de recherche
            filters: Dictionnaire de filtres à appliquer
            
        Returns:
            Liste filtrée des résultats
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            if self._match_filters(result, filters):
                filtered_results.append(result)
        
        self.log(LogLevel.DEBUG, f"Filtrage: {len(results)} -> {len(filtered_results)} résultats")
        return filtered_results
    
    def _match_filters(self, result: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Vérifie si un résultat correspond aux filtres.
        
        Args:
            result: Résultat de recherche
            filters: Filtres à appliquer
            
        Returns:
            True si le résultat correspond, False sinon
        """
        payload = result.get("payload", {})
        
        for key, value in filters.items():
            if not self._match_single_filter(payload, key, value):
                return False
        
        return True
    
    def _match_single_filter(self, payload: Dict[str, Any], key: str, value: Any) -> bool:
        """
        Vérifie un filtre individuel.
        
        Args:
            payload: Payload du document
            key: Clé du filtre
            value: Valeur du filtre
            
        Returns:
            True si le filtre correspond, False sinon
        """
        # Filtre de plage de dates
        if key == "upload_date_range" and isinstance(value, dict):
            date = payload.get("upload_date", "")
            if "start" in value and date < value["start"]:
                return False
            if "end" in value and date > value["end"]:
                return False
            return True
        
        # Filtres booléens pour la présence de listes
        if key.startswith("has_") and isinstance(value, bool) and value:
            list_name = key[4:]  # Retirer le préfixe "has_"
            list_items = payload.get(list_name, [])
            return bool(list_items)
        
        # Filtre exact
        return payload.get(key) == value
    
    def health_check(self) -> ServiceResponse:
        """Vérifie l'état de santé du filtreur de résultats"""
        return ServiceResponse.success_response({"status": "healthy", "service": "SearchResultFilter"})