"""
Tâche d'enrichissement LLM pour l'extraction d'entités et l'amélioration des métadonnées.
"""
import logging
from typing import Dict, Any, List
from ..ClassTraitement import Traitement
from ....services.llm_service import llm_service

logger = logging.getLogger(__name__)


class LLMEnrichmentTask(Traitement):
    """
    Tâche d'enrichissement utilisant un LLM pour extraire des entités
    et améliorer les métadonnées des documents.
    """
    
    def __init__(self):
        super().__init__("LLM Enrichment")
        logger.info("LLMEnrichmentTask initialisé")
    
    def prepare(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données pour l'enrichissement LLM.
        
        Args:
            data: Données brutes du document
            
        Returns:
            Données préparées pour l'enrichissement
        """
        return data
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute l'enrichissement LLM sur le document.
        
        Args:
            data: Données du document à enrichir
            
        Returns:
            Données enrichies avec les métadonnées LLM
        """
        try:
            logger.info(f"Début de l'enrichissement LLM pour: {data.get('filename', 'document inconnu')}")

            # Choisir le meilleur texte disponible: cleaned_text (prétraité) > extracted_text > content
            content = (
                data.get('cleaned_text')
                or data.get('extracted_text')
                or data.get('content')
                or ''
            )

            if not isinstance(content, str) or not content.strip():
                logger.warning("Aucun texte exploitable pour l'enrichissement LLM (cleaned_text/extracted_text/content)")
                data['llm_enrichment_status'] = 'not_performed'
                return data

            # Schéma d'extraction structuré attendu du LLM
            schema = {
                "llm_entities": [{"text": "string", "label": "string"}],
                "llm_keywords": ["string"],
                "llm_concepts": ["string"],
                "llm_summary": "string",
                "llm_document_type": "string",
                "llm_sentiment": "string",
                "llm_importance": "number"
            }

            response = llm_service.extract_structured_data(content, schema)

            # Si le service retourne un fallback/erreur, marquer comme non effectué
            if isinstance(response, dict) and response.get("error"):
                logger.warning("LLM a retourné une réponse de secours. Enrichissement non effectué.")
                data['llm_enrichment_status'] = 'not_performed'
                data['llm_enrichment_error'] = response.get('message', 'fallback')
                return data

            # Normaliser et injecter les champs attendus en sortie de pipeline
            data['llm_entities'] = response.get('llm_entities', []) or []
            data['llm_keywords'] = response.get('llm_keywords', []) or []
            data['llm_concepts'] = response.get('llm_concepts', []) or []
            data['llm_summary'] = response.get('llm_summary', '') or ''
            data['llm_document_type'] = response.get('llm_document_type', '') or ''
            data['llm_sentiment'] = response.get('llm_sentiment', 'neutre') or 'neutre'
            # importance en entier raisonnable 1..10
            importance = response.get('llm_importance', 5)
            try:
                importance = int(importance)
            except Exception:
                importance = 5
            data['llm_importance'] = max(1, min(10, importance))

            data['llm_enrichment_status'] = 'success'

            logger.info("Enrichissement LLM terminé avec succès")
            return data

        except Exception as e:
            logger.error(f"Erreur lors de l'enrichissement LLM: {e}")
            data['llm_enrichment_status'] = 'error'
            data['llm_enrichment_error'] = str(e)
            return data
    
    def _extract_entities(self, content: str) -> Dict[str, Any]:
        """
        Extrait les entités du contenu (version simplifiée).
        
        Args:
            content: Contenu textuel du document
            
        Returns:
            Dictionnaire des entités extraites
        """
        try:
            # Version simplifiée qui détecte quelques patterns basiques
            # Dans une version complète, ceci utiliserait un vrai LLM
            
            entities = {
                'has_emails': '@' in content and '.' in content,
                'has_phone_numbers': any(char.isdigit() for char in content) and ('tel' in content.lower() or 'phone' in content.lower()),
                'has_dates': any(word in content.lower() for word in ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                                                                       'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre',
                                                                       '2020', '2021', '2022', '2023', '2024']),
                'has_money_amounts': any(symbol in content for symbol in ['€', '$', 'EUR', 'USD']),
                'content_length': len(content),
                'word_count': len(content.split()),
                'language_detected': 'fr' if any(word in content.lower() for word in ['le', 'la', 'les', 'un', 'une', 'des']) else 'unknown'
            }
            
            logger.debug(f"Entités extraites: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction d'entités: {e}")
            return {
                'extraction_error': str(e),
                'content_length': len(content) if content else 0
            }