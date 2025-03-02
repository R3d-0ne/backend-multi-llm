import logging
import uuid
from fastapi import HTTPException

from .qdrant_service import qdrant_service
from ..models.settings import Settings

logger = logging.getLogger(__name__)

class SettingsService:
    def __init__(self):
        self.collection_name = "settings"
        # On suppose que la collection "messages" a déjà été créée dans la base de données.
        logger.info(f"Utilisation de la collection '{self.collection_name}' pour les settings.")

    def create_settings(self, settings_data: Settings) -> dict:
        """
        Crée un nouveau document de paramètres dans Qdrant.
        Chaque création génère un nouveau document avec un ID unique.
        """
        try:
            # Convertir l'instance Settings en dictionnaire
            payload = settings_data.model_dump() if hasattr(settings_data, "model_dump") else settings_data.dict()
            document_id = str(uuid.uuid4())
            dummy_vector = [0.0]
            qdrant_service.upsert_document(
                collection_name=self.collection_name,
                document_id=document_id,
                vector=dummy_vector,
                payload=payload
            )
            return {"message": "Paramètres créés avec succès", "document_id": document_id}
        except Exception as e:
            logger.error(f"Erreur lors de la création des paramètres : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la création des paramètres")

    def update_settings(self, settings_id: str, settings_data: Settings) -> dict:
        """
        Met à jour un document de paramètres existant dans Qdrant.
        Utilise l'ID fourni pour identifier le document à mettre à jour.
        """
        try:
            payload = settings_data.model_dump() if hasattr(settings_data, "model_dump") else settings_data.dict()
            document_id = settings_id
            dummy_vector = [0.0]
            qdrant_service.upsert_document(
                collection_name=self.collection_name,
                document_id=document_id,
                vector=dummy_vector,
                payload=payload
            )
            return {"message": "Paramètres mis à jour avec succès", "document_id": document_id}
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des paramètres : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour des paramètres")

    def get_settings_by_id(self, setting_id: str) -> dict:
        """
        Récupère les paramètres correspondant à l'ID fourni depuis la collection "settings".
        Retourne le payload du document, ou un dictionnaire vide si non trouvé.
        """
        try:
            result = qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=setting_id
            )
            if result and isinstance(result, list) and len(result) > 0:
                record = result[0]
                # Accès aux attributs selon la structure retournée par Qdrant
                if isinstance(record, dict):
                    return record.get("payload", {})
                else:
                    return record.payload
            return {}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des paramètres pour l'ID {setting_id} : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération des paramètres")

    def get_all_settings(self) -> list:
        """
        Récupère toutes les discussions (jusqu'à une limite de 1000) et les formate en liste de dictionnaires.
        """
        try:
            result, _ = qdrant_service.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            # settings = [
            #     {"id": point.id, "vector": point.vector, "payload": point.payload}
            #     for point in result
            # ]
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de toutes les settings: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération de toutes les settings")


# Instance singleton pour une utilisation centralisée
settings_service = SettingsService()
