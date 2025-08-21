import logging
import os
from typing import List, Dict, Any, Optional
from .qdrant_service import QdrantService
from .document_upload_service import DocumentUploadService

logger = logging.getLogger(__name__)

class DocumentService:
    """
    Service de gestion des documents.
    Permet de récupérer, lister et supprimer des documents en utilisant Qdrant.
    La création des documents reste dans DocumentUploadService.
    """

    def __init__(self):
        self.collection_name = "documents"
        self.qdrant_service = QdrantService()
        self.upload_service = DocumentUploadService()
        logger.info(f"Service de document initialisé avec la collection '{self.collection_name}'.")
        
        # Initialisation paresseuse de la collection (sera créée lors du premier usage)
        self._collection_initialized = False

    def _ensure_collection_exists(self):
        """
        S'assure que la collection existe, la crée si nécessaire.
        Méthode appelée paresseusement lors du premier accès.
        """
        if not self._collection_initialized:
            try:
                if not self.qdrant_service.collection_exists(self.collection_name):
                    self.qdrant_service.create_collection(
                        collection_name=self.collection_name,
                        vector_size=1536,  # Taille standard pour les embeddings
                        distance="Cosine"
                    )
                    logger.info(f"Collection '{self.collection_name}' créée.")
                self._collection_initialized = True
            except Exception as e:
                logger.warning(f"Impossible de vérifier/créer la collection: {e}")
                # Continue sans erreur pour permettre l'initialisation du service

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Récupère les informations d'un document à partir de son identifiant.
        
        :param document_id: Identifiant unique du document.
        :return: Dictionnaire contenant les informations du document.
        :raises ValueError: Si le document n'est pas trouvé.
        """
        self._ensure_collection_exists()
        try:
            # Récupérer les métadonnées depuis Qdrant
            records = self.qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            if not records:
                return {
                    "document_id": document_id,
                    "metadata": {}
                }
            
            # Extraire les métadonnées du premier enregistrement
            metadata = records[0].payload if records else {}
            
            return {
                "document_id": document_id,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document {document_id}: {e}")
            raise ValueError(f"Document non trouvé: {e}")

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Liste tous les documents disponibles directement depuis Qdrant.
        
        :return: Liste des documents avec leurs métadonnées.
        """
        self._ensure_collection_exists()
        try:
            # Récupérer directement les points depuis Qdrant
            response = self.qdrant_service.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Limite raisonnable pour éviter de surcharger la mémoire
                with_payload=True,
                with_vectors=False  # On n'a pas besoin des vecteurs pour le listing
            )
            
            # Extraire les points de la réponse
            points = response[0]
            
            # Transformer les points en documents
            documents = []
            for point in points:
                documents.append({
                    "document_id": point.id,
                    "metadata": point.payload if point.payload else {}
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Erreur lors du listage des documents depuis Qdrant: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Supprime un document de Qdrant.
        
        :param document_id: Identifiant unique du document.
        :return: True si la suppression a réussi.
        :raises ValueError: Si une erreur survient lors de la suppression.
        """
        self._ensure_collection_exists()
        try:
            # Supprimer les métadonnées de Qdrant
            self.qdrant_service.delete_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du document {document_id}: {e}")
            raise ValueError(f"Erreur lors de la suppression: {e}")

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Met à jour les métadonnées d'un document.
        
        :param document_id: Identifiant unique du document.
        :param metadata: Nouvelles métadonnées à associer au document.
        :return: Document mis à jour avec ses métadonnées.
        :raises ValueError: Si le document n'est pas trouvé.
        """
        self._ensure_collection_exists()
        try:
            # Mettre à jour les métadonnées dans Qdrant
            self.qdrant_service.update_document_metadata(
                collection_name=self.collection_name,
                document_id=document_id,
                new_metadata=metadata
            )
            
            return {
                "document_id": document_id,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métadonnées du document {document_id}: {e}")
            raise ValueError(f"Document non trouvé ou erreur de mise à jour: {e}")


# Instance du service de documents
document_service = DocumentService() 