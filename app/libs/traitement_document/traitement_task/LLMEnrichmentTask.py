"""
Tâche d'enrichissement LLM pour l'extraction d'entités et l'amélioration des métadonnées.
"""
import logging
from typing import Dict, Any, List
from ..ClassTraitement import Traitement

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
            
            # Récupérer le texte du document
            content = data.get('content', '')
            if not content:
                logger.warning("Aucun contenu trouvé pour l'enrichissement LLM")
                return data
            
            # Enrichir les métadonnées (version simplifiée pour éviter les dépendances)
            enriched_metadata = self._extract_entities(content)
            
            # Ajouter les métadonnées enrichies
            if 'metadata' not in data:
                data['metadata'] = {}
            
            data['metadata'].update(enriched_metadata)
            data['llm_enrichment_completed'] = True
            
            logger.info("Enrichissement LLM terminé avec succès")
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enrichissement LLM: {e}")
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