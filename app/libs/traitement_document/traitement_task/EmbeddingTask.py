import os
import logging
from typing import List, Dict, Any

from ..ClassTraitement import Traitement
from ....services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class EmbeddingTask(Traitement):
    def __init__(self):
        super().__init__("Embedding Task", max_retries=3, retry_delay=2)

    def prepare(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données pour la génération d'embeddings.
        Attend un dictionnaire contenant les résultats du prétraitement :
        - 'document_id': ID du document
        - 'cleaned_text': texte nettoyé
        - 'tokens': tokens du texte
        - 'tokens_no_stopwords': tokens sans stopwords
        - 'stemmed_tokens': tokens stemmatisés
        - 'lemmatized_tokens': tokens lemmatisés
        - 'named_entities_spacy': entités nommées extraites par spaCy
        - 'named_entities_flair': entités nommées extraites par Flair
        - 'named_entities_combined': entités nommées combinées
        """
        if not isinstance(data, dict):
            raise ValueError("EmbeddingTask attend un dictionnaire en entrée.")

        required_fields = [
            'document_id', 'cleaned_text', 'tokens',
            'tokens_no_stopwords', 'stemmed_tokens', 'lemmatized_tokens',
            'named_entities_spacy', 'named_entities_flair', 'named_entities_combined'
        ]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Le champ '{field}' est requis dans les données d'entrée.")

        logger.info(f"{self.name} - Données vérifiées pour le document {data['document_id']}")
        return data

    def execute(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère les embeddings pour les différentes versions du texte traité.
        """
        result = prepared_data.copy()
        document_id = prepared_data['document_id']

        try:
            # Génération des embeddings pour les différentes versions du texte
            embeddings = {
                'cleaned_text': embedding_service.get_embedding(prepared_data['cleaned_text']),
                'tokens': embedding_service.get_embedding(' '.join(prepared_data['tokens'])),
                'tokens_no_stopwords': embedding_service.get_embedding(' '.join(prepared_data['tokens_no_stopwords'])),
                'stemmed_tokens': embedding_service.get_embedding(' '.join(prepared_data['stemmed_tokens'])),
                'lemmatized_tokens': embedding_service.get_embedding(' '.join(prepared_data['lemmatized_tokens'])),
                'named_entities_spacy': embedding_service.get_embedding(' '.join([str(ent) for ent in prepared_data['named_entities_spacy']])),
                'named_entities_flair': embedding_service.get_embedding(' '.join([str(ent) for ent in prepared_data['named_entities_flair']])),
                'named_entities_combined': embedding_service.get_embedding(' '.join([str(ent) for ent in prepared_data['named_entities_combined']]))
            }
            
            # Ajout des embeddings au résultat
            result['embeddings'] = embeddings
            result['embedding_status'] = 'success'

            # S'assurer que filename et upload_date sont préservés
            result['filename'] = prepared_data.get('filename')
            result['upload_date'] = prepared_data.get('upload_date')

            logger.info(f"{self.name} - Embeddings générés pour le document {document_id}")

        except Exception as e:
            logger.error(f"{self.name} - Erreur lors de la génération des embeddings : {e}")
            result['embedding_status'] = 'error'
            result['error'] = str(e)
            raise

        return result 