import logging
from typing import Dict, Any
import numpy as np
import uuid
import os
from datetime import datetime

from ..ClassTraitement import Traitement
from ....services.qdrant_service import qdrant_service
from ....libs.functions.global_functions import (
    convert_numpy_types,
    encode_image_to_base64,
    create_temp_directory,
    cleanup_temp_directory,
    create_sparse_vector
)

logger = logging.getLogger(__name__)

class StorageTask(Traitement):
    def __init__(self):
        super().__init__("Storage Task", max_retries=3, retry_delay=2)
        self.temp_dir = None
        self.use_hybrid_storage = True  # Activer le stockage hybride par défaut

    def prepare(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données pour le stockage dans Qdrant.
        Attend un dictionnaire contenant toutes les données traitées :
        - 'document_id': ID du document
        - 'embeddings': dictionnaire des embeddings générés
        - 'cleaned_text': texte nettoyé
        - 'tokens': tokens du texte
        - 'filename': nom du fichier original
        - 'upload_date': date de dépôt
        - etc.
        """
        if not isinstance(data, dict):
            raise ValueError("StorageTask attend un dictionnaire en entrée.")

        required_fields = ['document_id', 'embeddings']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Le champ '{field}' est requis dans les données d'entrée.")

        # Vérification des champs optionnels avec valeurs par défaut
        if 'filename' not in data:
            logger.warning("Le champ 'filename' n'est pas présent dans les données")
            data['filename'] = ''

        if 'upload_date' not in data:
            logger.warning("Le champ 'upload_date' n'est pas présent, utilisation de la date actuelle")
            data['upload_date'] = datetime.now().isoformat()

        # Vérifier que l'embedding du texte nettoyé existe
        if 'cleaned_text' not in data['embeddings']:
            raise ValueError("L'embedding du texte nettoyé est requis pour le stockage.")

        # Créer un dossier temporaire pour le traitement
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        temp_base = os.path.join(base_path, "temp")
        os.makedirs(temp_base, exist_ok=True)
        self.temp_dir = create_temp_directory(temp_base, f"storage_{data['document_id']}_")

        logger.info(f"{self.name} - Données vérifiées pour le document {data['document_id']}")
        return data

    def execute(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stocke toutes les données traitées dans Qdrant.
        Utilise l'embedding du texte nettoyé comme vecteur principal pour la recherche.
        """
        result = prepared_data.copy()
        document_id = prepared_data['document_id']
        embeddings = prepared_data['embeddings']

        try:
            # Vérification et traitement du filename et upload_date
            filename = prepared_data.get('filename')
            if not filename or filename.strip() == '':
                # Si le filename est vide, on essaie de le récupérer depuis image_paths
                if prepared_data.get('image_paths') and len(prepared_data['image_paths']) > 0:
                    filename = os.path.basename(prepared_data['image_paths'][0])
                else:
                    filename = f"document_{document_id}"
                logger.info(f"{self.name} - Utilisation du nom de fichier par défaut: {filename}")

            upload_date = prepared_data.get('upload_date')
            if not upload_date or upload_date.strip() == '':
                upload_date = datetime.now().isoformat()
                logger.info(f"{self.name} - Utilisation de la date actuelle pour upload_date: {upload_date}")

            # Convertir les images en base64 si nécessaire
            image_data = []
            for image_path in prepared_data.get('image_paths', []):
                if os.path.exists(image_path):
                    encoded_image = encode_image_to_base64(image_path)
                    if encoded_image:
                        image_data.append({
                            'filename': os.path.basename(image_path),
                            'data': encoded_image
                        })
                        logger.info(f"{self.name} - Image convertie en base64: {os.path.basename(image_path)}")

            # Si aucune image n'a été convertie, vérifier image_data existant
            if not image_data:
                image_data = prepared_data.get('image_data', [])

            # Vérifier si nous utilisons le stockage hybride
            if self.use_hybrid_storage:
                # 1. Préparer les vecteurs pour le stockage hybride
                # Extraction et traitement des embeddings pour chaque type
                dense_vector = embeddings.get('cleaned_text')
                if hasattr(dense_vector, "tolist"):
                    dense_vector = dense_vector.tolist()
                elif isinstance(dense_vector, list) and dense_vector and isinstance(dense_vector[0], list):
                    dense_vector = dense_vector[0]
                
                # Sélectionner le meilleur vecteur d'entités disponible
                # Priorité: 1. Entités LLM, 2. Entités BERT, 3. Entités combinées
                entity_vector = None
                entity_source = ""
                
                if 'llm_entities' in embeddings:
                    entity_vector = embeddings.get('llm_entities')
                    entity_source = "llm_entities"
                    llm_entities_count = len(prepared_data.get('llm_entities', []))
                    logger.info(f"{self.name} - Utilisation des entités LLM ({llm_entities_count} entités) pour le vecteur d'entités")
                elif 'named_entities_bert' in embeddings:
                    entity_vector = embeddings.get('named_entities_bert')
                    entity_source = "named_entities_bert"
                    logger.info(f"{self.name} - Utilisation des entités BERT pour le vecteur d'entités")
                else:
                    entity_vector = embeddings.get('named_entities_combined')
                    entity_source = "named_entities_combined"
                    logger.info(f"{self.name} - Utilisation des entités combinées pour le vecteur d'entités")
                
                logger.info(f"{self.name} - Source du vecteur d'entités: {entity_source}")
                
                if hasattr(entity_vector, "tolist"):
                    entity_vector = entity_vector.tolist()
                elif isinstance(entity_vector, list) and entity_vector and isinstance(entity_vector[0], list):
                    entity_vector = entity_vector[0]
                
                # Création du vecteur épars à partir des tokens sans stopwords ou mots-clés LLM
                tokens_no_stopwords = prepared_data.get('tokens_no_stopwords', [])
                
                # Utiliser les mots-clés LLM si disponibles pour améliorer le vecteur sparse
                llm_keywords = prepared_data.get('llm_keywords', [])
                if llm_keywords:
                    tokens_no_stopwords = tokens_no_stopwords + llm_keywords
                    logger.info(f"{self.name} - Ajout de {len(llm_keywords)} mots-clés LLM au vecteur sparse: {', '.join(llm_keywords)}")
                
                sparse_vector = create_sparse_vector(tokens_no_stopwords, vocab_size=10000)
                
                # 2. Structure des vecteurs pour le stockage hybride
                vectors = {
                    "dense": dense_vector,
                    "entity": entity_vector,
                    "sparse": sparse_vector
                }
                
                # 3. Préparation du payload (comme avant)
                payload = {
                    'document_id': document_id,
                    'filename': filename,
                    'upload_date': upload_date,
                    'cleaned_text': prepared_data.get('cleaned_text', ''),
                    'tokens': prepared_data.get('tokens', []),
                    'tokens_no_stopwords': prepared_data.get('tokens_no_stopwords', []),
                    'stemmed_tokens': prepared_data.get('stemmed_tokens', []),
                    'lemmatized_tokens': prepared_data.get('lemmatized_tokens', []),
                    'named_entities_bert': prepared_data.get('named_entities_bert', []),
                    'phone_numbers': prepared_data.get('phone_numbers', []),
                    'emails': prepared_data.get('emails', []),
                    'money_amounts': prepared_data.get('money_amounts', []),
                    'dates': prepared_data.get('dates', []),
                    'percentages': prepared_data.get('percentages', []),
                    'has_tables': prepared_data.get('has_tables', False),
                    'tables_count': prepared_data.get('tables_count', 0),
                    'pages_with_tables': prepared_data.get('pages_with_tables', []),
                    'details_tables': prepared_data.get('details_tables', []),
                    'images': image_data,
                    
                    # Données d'enrichissement LLM
                    'llm_enrichment_status': prepared_data.get('llm_enrichment_status', 'not_performed'),
                    'llm_entities': prepared_data.get('llm_entities', []),
                    'llm_concepts': prepared_data.get('llm_concepts', []),
                    'llm_summary': prepared_data.get('llm_summary', ''),
                    'llm_keywords': prepared_data.get('llm_keywords', []),
                    'llm_document_type': prepared_data.get('llm_document_type', ''),
                    'llm_sentiment': prepared_data.get('llm_sentiment', 'neutre'),
                    'llm_importance': prepared_data.get('llm_importance', 5)
                }
                
                # 4. Stockage hybride dans Qdrant
                qdrant_service.upsert_hybrid_document(
                    collection_name="documents",
                    document_id=document_id,
                    vectors=vectors,
                    payload=payload
                )
                logger.info(f"{self.name} - Document {document_id} stocké avec l'approche hybride")
                
            else:
                # Ancien code de stockage avec un seul vecteur principal
                # Extraire et convertir l'embedding du texte nettoyé pour le vecteur principal
                main_vector = embeddings['cleaned_text']
                if hasattr(main_vector, "tolist"):
                    main_vector = main_vector.tolist()
                elif isinstance(main_vector, list) and main_vector and isinstance(main_vector[0], list):
                    main_vector = main_vector[0]

                # Convertir les autres embeddings pour le payload
                embeddings_dict = {}
                for version, embedding in embeddings.items():
                    if version != 'cleaned_text':  # On exclut l'embedding principal
                        if hasattr(embedding, "tolist"):
                            embeddings_dict[version] = embedding.tolist()
                        elif isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                            embeddings_dict[version] = embedding[0]
                        else:
                            embeddings_dict[version] = embedding

                # Convertir les images en base64
                image_data = []
                for image_path in prepared_data.get('image_paths', []):
                    if os.path.exists(image_path):
                        encoded_image = encode_image_to_base64(image_path)
                        if encoded_image:
                            image_data.append({
                                'filename': os.path.basename(image_path),
                                'data': encoded_image
                            })
                
                # Préparation du payload avec toutes les métadonnées
                payload = {
                    'document_id': document_id,
                    'filename': filename,  # Nom du fichier traité
                    'upload_date': upload_date,  # Date de dépôt traitée
                    'embeddings': embeddings_dict,  # Les autres embeddings dans le payload
                    'cleaned_text': prepared_data.get('cleaned_text', ''),
                    'tokens': prepared_data.get('tokens', []),
                    'tokens_no_stopwords': prepared_data.get('tokens_no_stopwords', []),
                    'stemmed_tokens': prepared_data.get('stemmed_tokens', []),
                    'lemmatized_tokens': prepared_data.get('lemmatized_tokens', []),
                    'named_entities_bert': prepared_data.get('named_entities_bert', []),
                    'phone_numbers': prepared_data.get('phone_numbers', []),
                    'emails': prepared_data.get('emails', []),
                    'money_amounts': prepared_data.get('money_amounts', []),
                    'dates': prepared_data.get('dates', []),
                    'percentages': prepared_data.get('percentages', []),
                    'has_tables': prepared_data.get('has_tables', False),
                    'tables_count': prepared_data.get('tables_count', 0),
                    'pages_with_tables': prepared_data.get('pages_with_tables', []),
                    'details_tables': prepared_data.get('details_tables', []),
                    'images': image_data,  # Images encodées en base64
                    
                    # Données d'enrichissement LLM
                    'llm_enrichment_status': prepared_data.get('llm_enrichment_status', 'not_performed'),
                    'llm_entities': prepared_data.get('llm_entities', []),
                    'llm_concepts': prepared_data.get('llm_concepts', []),
                    'llm_summary': prepared_data.get('llm_summary', ''),
                    'llm_keywords': prepared_data.get('llm_keywords', []),
                    'llm_document_type': prepared_data.get('llm_document_type', ''),
                    'llm_sentiment': prepared_data.get('llm_sentiment', 'neutre'),
                    'llm_importance': prepared_data.get('llm_importance', 5)
                }

                # Stockage dans Qdrant (un seul point par document)
                qdrant_service.upsert_document(
                    collection_name="documents",
                    document_id=document_id,
                    vector=main_vector,
                    payload=payload
                )
                logger.info(f"{self.name} - Document {document_id} stocké avec l'approche standard")

            # Ajouter filename et upload_date au résultat
            result['filename'] = filename
            result['upload_date'] = upload_date
            result['storage_status'] = 'success'
            # Ne pas inclure les embeddings dans le résultat pour éviter les problèmes de sérialisation
            if 'embeddings' in result:
                del result['embeddings']

        except Exception as e:
            logger.error(f"{self.name} - Erreur lors du stockage dans Qdrant : {e}")
            result['storage_status'] = 'error'
            result['error'] = str(e)
            raise

        finally:
            # Nettoyage du dossier temporaire
            if self.temp_dir and os.path.exists(self.temp_dir):
                cleanup_temp_directory(self.temp_dir)
                self.temp_dir = None

        return convert_numpy_types(result) 