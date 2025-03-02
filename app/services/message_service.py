import uuid
import logging
from typing import List, Dict, Any
from fastapi import HTTPException

from .embedding_service import embedding_service
from .qdrant_service import qdrant_service
from ..models.message import Message

logger = logging.getLogger(__name__)


class MessageService:
    def __init__(self):
        self.collection_name = "messages"
        # On suppose que la collection "messages" a déjà été créée dans la base de données.
        logger.info(f"Utilisation de la collection '{self.collection_name}' pour les messages.")

    def send_message(self, message_data: Message) -> dict:
        """
        Enregistre un message dans Qdrant en tant que nouveau document.
        Calcule l'embedding à partir du contenu du message.
        """
        try:
            # Convertir l'instance Message en dictionnaire
            payload = message_data.model_dump() if hasattr(message_data, "model_dump") else message_data.dict()

            # Calculer l'embedding à partir du texte du message
            vector = embedding_service.get_embedding(message_data.text)
            # Conversion si nécessaire (ex: numpy.ndarray → liste)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            # Si le vecteur est imbriqué (liste de listes), aplatir en prenant la première sous-liste
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]

            # Générer un identifiant unique pour ce message (si non déjà présent)
            message_id = payload.get("id", str(uuid.uuid4()))

            # Insérer le message dans Qdrant
            qdrant_service.upsert_document(
                collection_name=self.collection_name,
                document_id=message_id,
                vector=vector,
                payload=payload
            )
            return {"message": "Message envoyé et enregistré", "id": message_id}
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du message : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'envoi du message")

    def update_message(self, message_id: str, new_data: Dict[str, Any]) -> dict:
        """
        Met à jour les métadonnées (payload) d'un message existant dans la collection "messages".
        `new_data` est un dictionnaire contenant les clés à mettre à jour.
        """
        try:
            qdrant_service.update_document_metadata(
                collection_name=self.collection_name,
                document_id=message_id,
                new_metadata=new_data
            )
            return {"message": "Message mis à jour", "id": message_id}
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du message {message_id} : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour du message")

    def delete_message(self, message_id: str) -> dict:
        """
        Supprime un message de la collection "messages" en utilisant son identifiant.
        """
        try:
            qdrant_service.delete_document(
                collection_name=self.collection_name,
                document_id=message_id
            )
            return {"message": "Message supprimé", "id": message_id}
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du message {message_id} : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression du message")

    def get_messages_by_discussion(self, discussion_id: str) -> List[Dict[str, Any]]:
        """
        Récupère tous les messages associés à une discussion en filtrant sur le champ 'discussion_id'
        présent dans le payload.
        """
        try:
            # Définir le filtre pour récupérer les messages liés à la discussion
            filter_payload = {
                "must": [
                    {"key": "discussion_id", "match": {"value": discussion_id}}
                ]
            }
            result = qdrant_service.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_payload,  #
                limit=1000  #
            )
            # Gérer les différents formats de réponse
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, dict):
                points = result.get("result", [])
            elif isinstance(result, list):
                points = result
            else:
                raise ValueError(f"Type de résultat inattendu: {type(result)}")

            messages = [{"id": point.id, **point.payload} for point in points]
            return messages
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des messages pour la discussion {discussion_id}: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération des messages")


# Instance singleton pour une utilisation centralisée
message_service = MessageService()
