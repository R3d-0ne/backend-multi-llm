from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
import logging

from .embedding_service import embedding_service
from .qdrant_service import qdrant_service
from ..models.discussion import Discussion

logger = logging.getLogger(__name__)
router = APIRouter()



class DiscussionService:
    COLLECTION_NAME = "discussions"

    def __init__(self):
        self.collection_name = "discussions"
        # On suppose que la collection "messages" a déjà été créée dans la base de données.
        logger.info(f"Utilisation de la collection '{self.collection_name}' pour les discussions.")

    def add_discussion(self, discussion: Discussion) -> dict:
        """
        Ajoute une discussion (texte seul) dans la collection.
        L'embedding du texte est calculé et stocké avec le payload.
        """
        try:
            # Accepte soit un objet Discussion, soit une chaîne (cas inhabituel)
            discussion_text = discussion if isinstance(discussion, str) else discussion.title

            # Calcul de l'embedding pour la discussion
            embedding = embedding_service.get_embedding(discussion_text)
            discussion_id = str(uuid.uuid4())
            
            # Créer le payload avec le titre
            payload = {"title": discussion_text}

            qdrant_service.upsert_document(
                collection_name=self.COLLECTION_NAME,
                document_id=discussion_id,
                vector=embedding,
                payload=payload
            )
            
            # Retourner l'ID de la discussion créée
            return {"id": discussion_id, "message": "Discussion ajoutée avec succès"}
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de la discussion: {e}")
            raise HTTPException(status_code=500, detail="Échec de l'ajout de la discussion")

    def get_discussion(self, discussion_id: str) -> dict:
        """
        Récupère une discussion par son ID.
        Retourne l'ID, le vecteur et le payload (contenant le texte).
        """
        try:
            result = qdrant_service.get_document(
                collection_name=self.COLLECTION_NAME,
                document_id=discussion_id
            )
            if not result:
                raise HTTPException(status_code=404, detail="Discussion introuvable")
            record = result[0]  # On suppose qu'il y a un seul record par ID
            return {"id": record.id, "vector": record.vector, "payload": record.payload}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la discussion: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération de la discussion")

    def get_all_discussions(self) -> list:
        """
        Récupère toutes les discussions (jusqu'à une limite de 1000) et les formate en liste de dictionnaires.
        """
        try:
            result, _ = qdrant_service.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=1000,
                with_payload=True,
                with_vectors=True
            )
            discussions = [
                {"id": point.id, "vector": point.vector, "payload": point.payload}
                for point in result
            ]
            return discussions
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de toutes les discussions: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération de toutes les discussions")

    def update_discussion(self, discussion_id: str, discussion: Discussion) -> dict:
        """
        Met à jour une discussion existante en régénérant l'embedding et en mettant à jour le payload.
        """
        try:
            embedding = embedding_service.get_embedding(discussion.title)
            payload = {"title": discussion.title}
            qdrant_service.upsert_document(
                collection_name=self.COLLECTION_NAME,
                document_id=discussion_id,
                vector=embedding,
                payload=payload
            )
            return {"id": discussion_id, "message": "Discussion mise à jour avec succès"}
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la discussion: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour de la discussion")

    def delete_discussion(self, discussion_id: str) -> dict:
        """
        Supprime une discussion identifiée par son ID.
        """
        try:
            qdrant_service.delete_document(
                collection_name=self.COLLECTION_NAME,
                document_id=discussion_id
            )
            return {"id": discussion_id, "message": "Discussion supprimée avec succès"}
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la discussion: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression de la discussion")

    def search_similar_discussions(self, discussion_id: str, limit: int = 3) -> list:
        """
        Recherche des discussions similaires dans la collection "discussions" à partir du texte de la discussion donnée.
        Exclut la discussion courante des résultats.
        """
        try:
            # Récupérer la discussion actuelle
            current = self.get_discussion(discussion_id)
            discussion_text = current.get("payload", {}).get("text", "")
            if not discussion_text:
                return []

            # Calcul de l'embedding à partir du texte
            query_vector = embedding_service.get_embedding(discussion_text)

            # Recherche des discussions similaires dans la collection "discussions"
            similar = qdrant_service.search_similar(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit + 1  # +1 pour pouvoir filtrer la discussion courante
            )
            # Filtrer la discussion courante
            similar = [d for d in similar if d["id"] != discussion_id]
            return similar[:limit]
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de discussions similaires: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la recherche de discussions similaires")
   
    # def append_message(self, discussion_id: str, role: str, message: str) -> dict:
        # """
        # Ajoute un nouveau message à une discussion existante.
        # Le nouveau message (avec rôle 'user' ou 'assistant') est ajouté à la liste des messages,
        # le texte global de la discussion est reconstruit et l'embedding est recalculé.
        # """
        # try:
        #     # Récupérer la discussion existante
        #     current = self.get_discussion(discussion_id)
        #     if not current:
        #         raise HTTPException(status_code=404, detail="Discussion introuvable")

        #     payload = current.get("payload", {})
        #     messages = payload.get("messages", [])
        #     # Ajouter le nouveau message
        #     messages.append({
        #         "role": role,
        #         "message": message,
        #         "timestamp": datetime.utcnow().isoformat()
        #     })
        #     # Reconstruire le texte global à partir de tous les messages
        #     concatenated_text = "\n".join([m["message"] for m in messages])
        #     # Recalculer l'embedding pour la discussion mise à jour
        #     new_embedding = embedding_service.get_embedding(concatenated_text)
        #     new_payload = {
        #         "text": concatenated_text,
        #         "messages": messages
        #     }
        #     qdrant_service.upsert_document(
        #         collection_name=self.COLLECTION_NAME,
        #         document_id=discussion_id,
        #         vector=new_embedding,
        #         payload=new_payload
        #     )
        #     return {"id": discussion_id, "message": "Message ajouté avec succès", "payload": new_payload}
        # except Exception as e:
        #     logger.error(f"Erreur lors de l'ajout du message: {e}")
        #     raise HTTPException(status_code=500, detail="Erreur lors de l'ajout du message")


# Instance unique du service
discussions_service = DiscussionService()
