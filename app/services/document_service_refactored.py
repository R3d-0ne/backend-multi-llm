"""
Service de gestion des documents refactorisé avec validation et gestion d'erreurs améliorées.
"""
import logging
from typing import List, Dict, Any, Optional
from .base_service import BaseService, LogLevel, ServiceResponse, ValidationError
from .config_service import config_service
from .qdrant_service import QdrantService
from .document_upload_service import DocumentUploadService


class DocumentServiceRefactored(BaseService):
    """
    Service de gestion des documents refactorisé avec validation robuste.
    Permet de récupérer, lister et supprimer des documents en utilisant Qdrant.
    """

    def __init__(self):
        super().__init__("DocumentService")
        self.collection_name = config_service.search.default_collection
        self.qdrant_service = QdrantService()
        self.upload_service = DocumentUploadService()
        
        self.log(LogLevel.INFO, f"Service de document initialisé avec la collection '{self.collection_name}'")
        
        # Vérifier et créer la collection si nécessaire
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """
        S'assure que la collection par défaut existe.
        """
        try:
            if not self.qdrant_service.collection_exists(self.collection_name):
                self.log(LogLevel.INFO, f"Création de la collection '{self.collection_name}'")
                self.qdrant_service.create_collection(
                    collection_name=self.collection_name,
                    vector_size=1536,  # Taille standard pour les embeddings
                    distance="Cosine"
                )
                self.log(LogLevel.INFO, f"Collection '{self.collection_name}' créée avec succès")
        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors de la création de la collection: {str(e)}")

    def get_document(self, document_id: str) -> ServiceResponse:
        """
        Récupère les informations d'un document à partir de son identifiant.
        
        Args:
            document_id: Identifiant unique du document
            
        Returns:
            ServiceResponse contenant les informations du document
        """
        # Validation
        if not document_id or not isinstance(document_id, str):
            return ServiceResponse.error_response(
                "ID de document invalide",
                "INVALID_DOCUMENT_ID",
                {"provided_id": document_id}
            )
        
        def get_operation():
            self.log(LogLevel.DEBUG, f"Récupération du document: {document_id}")
            
            # Récupérer les métadonnées depuis Qdrant
            records = self.qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            if not records:
                return {
                    "document_id": document_id,
                    "metadata": {},
                    "found": False,
                    "message": "Document non trouvé"
                }
            
            # Extraire les métadonnées du premier enregistrement
            metadata = records[0].payload if records else {}
            
            return {
                "document_id": document_id,
                "metadata": metadata,
                "found": True,
                "vector_available": hasattr(records[0], 'vector') and records[0].vector is not None
            }
        
        return self.safe_execute(
            get_operation,
            f"Erreur lors de la récupération du document {document_id}"
        )

    def list_documents(self, limit: int = 1000, offset: int = 0, filter_criteria: Dict[str, Any] = None) -> ServiceResponse:
        """
        Liste tous les documents disponibles avec pagination et filtrage optionnel.
        
        Args:
            limit: Nombre maximum de documents à retourner
            offset: Décalage pour la pagination
            filter_criteria: Critères de filtrage optionnels
            
        Returns:
            ServiceResponse contenant la liste des documents
        """
        # Validation
        if limit <= 0 or limit > 10000:
            return ServiceResponse.error_response(
                "Limite invalide (doit être entre 1 et 10000)",
                "INVALID_LIMIT",
                {"provided_limit": limit}
            )
        
        if offset < 0:
            return ServiceResponse.error_response(
                "Offset invalide (doit être >= 0)",
                "INVALID_OFFSET",
                {"provided_offset": offset}
            )
        
        def list_operation():
            self.log(LogLevel.DEBUG, f"Listage des documents: limite={limit}, offset={offset}")
            
            # Récupérer directement les points depuis Qdrant
            response = self.qdrant_service.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False  # On n'a pas besoin des vecteurs pour le listing
            )
            
            # Extraire les points de la réponse
            points = response[0] if response else []
            
            # Transformer les points en documents
            documents = []
            for point in points:
                doc_data = {
                    "document_id": str(point.id),
                    "metadata": point.payload if point.payload else {}
                }
                
                # Appliquer les filtres si fournis
                if filter_criteria is None or self._matches_filter(doc_data["metadata"], filter_criteria):
                    documents.append(doc_data)
            
            self.log(LogLevel.INFO, f"Documents listés: {len(documents)} trouvés")
            
            return {
                "documents": documents,
                "total_returned": len(documents),
                "limit": limit,
                "offset": offset,
                "has_more": len(points) == limit  # Indication qu'il pourrait y avoir plus de documents
            }
        
        return self.safe_execute(
            list_operation,
            "Erreur lors du listage des documents"
        )

    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """
        Vérifie si un document correspond aux critères de filtrage.
        
        Args:
            metadata: Métadonnées du document
            filter_criteria: Critères de filtrage
            
        Returns:
            True si le document correspond aux critères
        """
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            
            # Comparaison exacte pour les valeurs simples
            if isinstance(value, (str, int, float, bool)):
                if metadata[key] != value:
                    return False
            
            # Comparaison pour les listes (contient au moins un élément)
            elif isinstance(value, list):
                if not any(item in metadata.get(key, []) for item in value):
                    return False
        
        return True

    def delete_document(self, document_id: str) -> ServiceResponse:
        """
        Supprime un document de Qdrant.
        
        Args:
            document_id: Identifiant unique du document
            
        Returns:
            ServiceResponse confirmant la suppression
        """
        # Validation
        if not document_id or not isinstance(document_id, str):
            return ServiceResponse.error_response(
                "ID de document invalide",
                "INVALID_DOCUMENT_ID",
                {"provided_id": document_id}
            )
        
        def delete_operation():
            self.log(LogLevel.INFO, f"Suppression du document: {document_id}")
            
            # Vérifier que le document existe
            existing = self.qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            if not existing:
                return {
                    "document_id": document_id,
                    "deleted": False,
                    "message": "Document non trouvé"
                }
            
            # Supprimer le document de Qdrant
            self.qdrant_service.delete_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            self.log(LogLevel.INFO, f"Document supprimé avec succès: {document_id}")
            
            return {
                "document_id": document_id,
                "deleted": True,
                "message": "Document supprimé avec succès"
            }
        
        return self.safe_execute(
            delete_operation,
            f"Erreur lors de la suppression du document {document_id}"
        )

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> ServiceResponse:
        """
        Met à jour les métadonnées d'un document.
        
        Args:
            document_id: Identifiant unique du document
            metadata: Nouvelles métadonnées à associer au document
            
        Returns:
            ServiceResponse contenant le document mis à jour
        """
        # Validation
        if not document_id or not isinstance(document_id, str):
            return ServiceResponse.error_response(
                "ID de document invalide",
                "INVALID_DOCUMENT_ID",
                {"provided_id": document_id}
            )
        
        if not isinstance(metadata, dict):
            return ServiceResponse.error_response(
                "Métadonnées invalides (doit être un dictionnaire)",
                "INVALID_METADATA",
                {"provided_metadata": str(metadata)}
            )
        
        def update_operation():
            self.log(LogLevel.INFO, f"Mise à jour des métadonnées du document: {document_id}")
            
            # Vérifier que le document existe
            existing = self.qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=document_id
            )
            
            if not existing:
                raise ValidationError(f"Document non trouvé: {document_id}", "document_id", document_id)
            
            # Mettre à jour les métadonnées dans Qdrant
            self.qdrant_service.update_document_metadata(
                collection_name=self.collection_name,
                document_id=document_id,
                new_metadata=metadata
            )
            
            self.log(LogLevel.INFO, f"Métadonnées mises à jour avec succès pour: {document_id}")
            
            return {
                "document_id": document_id,
                "metadata": metadata,
                "updated": True,
                "message": "Métadonnées mises à jour avec succès"
            }
        
        return self.safe_execute(
            update_operation,
            f"Erreur lors de la mise à jour des métadonnées du document {document_id}"
        )

    def get_document_statistics(self) -> ServiceResponse:
        """
        Obtient des statistiques sur les documents dans la collection.
        
        Returns:
            ServiceResponse contenant les statistiques
        """
        def stats_operation():
            self.log(LogLevel.DEBUG, "Calcul des statistiques des documents")
            
            # Obtenir des informations sur la collection
            collection_info = self.qdrant_service.client.get_collection(self.collection_name)
            
            # Compter les documents par type de fichier
            response = self.qdrant_service.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Limite élevée pour avoir un aperçu complet
                with_payload=True,
                with_vectors=False
            )
            
            points = response[0] if response else []
            
            # Analyser les métadonnées
            file_types = {}
            upload_dates = []
            total_documents = len(points)
            
            for point in points:
                payload = point.payload or {}
                
                # Compter par type de fichier
                filename = payload.get('filename', '')
                if filename:
                    extension = filename.split('.')[-1].lower() if '.' in filename else 'unknown'
                    file_types[extension] = file_types.get(extension, 0) + 1
                
                # Collecter les dates
                upload_date = payload.get('upload_date')
                if upload_date:
                    upload_dates.append(upload_date)
            
            return {
                "total_documents": total_documents,
                "collection_name": self.collection_name,
                "file_types": file_types,
                "collection_info": {
                    "status": collection_info.status,
                    "vectors_count": collection_info.vectors_count,
                    "indexed_vectors_count": collection_info.indexed_vectors_count
                },
                "upload_dates_sample": sorted(upload_dates)[-10:] if upload_dates else []
            }
        
        return self.safe_execute(
            stats_operation,
            "Erreur lors du calcul des statistiques des documents"
        )

    def search_documents(self, search_criteria: Dict[str, Any], limit: int = 100) -> ServiceResponse:
        """
        Recherche des documents basés sur des critères spécifiques.
        
        Args:
            search_criteria: Critères de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            ServiceResponse contenant les documents trouvés
        """
        # Validation
        if not isinstance(search_criteria, dict) or not search_criteria:
            return ServiceResponse.error_response(
                "Critères de recherche invalides",
                "INVALID_SEARCH_CRITERIA"
            )
        
        def search_operation():
            self.log(LogLevel.DEBUG, f"Recherche de documents avec critères: {search_criteria}")
            
            # Pour l'instant, utiliser la méthode de listage avec filtrage
            # Dans une version future, on pourrait utiliser les filtres natifs de Qdrant
            list_response = self.list_documents(limit=limit, filter_criteria=search_criteria)
            
            if not list_response.success:
                raise Exception(list_response.error)
            
            documents = list_response.data.get("documents", [])
            
            return {
                "documents": documents,
                "total_found": len(documents),
                "search_criteria": search_criteria,
                "limit": limit
            }
        
        return self.safe_execute(
            search_operation,
            "Erreur lors de la recherche de documents"
        )

    def health_check(self) -> ServiceResponse:
        """
        Vérifie l'état de santé du service de documents.
        
        Returns:
            ServiceResponse avec l'état de santé
        """
        try:
            # Vérifier la connexion à Qdrant
            collection_exists = self.qdrant_service.collection_exists(self.collection_name)
            
            # Tenter une opération simple
            stats_response = self.get_document_statistics()
            
            health_data = {
                "status": "healthy",
                "service": "DocumentService",
                "collection_name": self.collection_name,
                "collection_exists": collection_exists,
                "qdrant_accessible": True,
                "statistics_accessible": stats_response.success
            }
            
            if stats_response.success:
                health_data["document_count"] = stats_response.data.get("total_documents", 0)
            
            return ServiceResponse.success_response(health_data)
            
        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur lors du test de santé: {str(e)}",
                "HEALTH_CHECK_ERROR",
                {"collection_name": self.collection_name}
            )


# Instance du service de documents refactorisé
document_service_refactored = DocumentServiceRefactored()