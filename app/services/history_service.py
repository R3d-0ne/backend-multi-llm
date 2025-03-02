from datetime import datetime
import uuid
import logging

from .embedding_service import embedding_service
from .qdrant_service import qdrant_service

logger = logging.getLogger(__name__)


class HistoryService:
    # def __init__(self):
    #     self.collection_name = "history"
    #     # Crée ou recrée la collection "history" dans Qdrant.
    #     # On suppose ici que la dimension des embeddings est 384.
    #     try:
    #         qdrant_service.create_collection(
    #             collection_name=self.collection_name,
    #             vector_size=384,
    #             distance="Cosine"
    #         )
    #     except Exception as e:
    #         logger.warning(f"Collection '{self.collection_name}' existante ou non recréable : {e}")

    def add_history(self, discussion_id: str, question: str, response: str) -> str:
        """
        Ajoute un échange (question/réponse) dans Qdrant et retourne son ID.
        Le vecteur est calculé à partir de la concaténation de la question et de la réponse.
        """
        history_id = str(uuid.uuid4())
        history_entry = {
            "discussion_id": discussion_id,
            "question": question,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        # Combine question et réponse pour générer l'embedding
        text_for_embedding = f"Q: {question} | A: {response}"
        embedding = embedding_service.get_embedding(text_for_embedding)

        # Utilise upsert_document pour enregistrer le document dans Qdrant
        qdrant_service.upsert_document(
            collection_name=self.collection_name,
            document_id=history_id,
            vector=embedding,
            payload=history_entry
        )
        return history_id

    def get_history_by_discussion(self, discussion_id: str) -> list:
        """
        Récupère l'historique d'une discussion spécifique depuis Qdrant.
        Utilise un filtre sur le payload pour ne récupérer que les documents
        dont le champ 'discussion_id' correspond.
        """
        # Définition du filtre pour la recherche (en tant que dict)
        filter_payload = {
            "must": [
                {"key": "discussion_id", "match": {"value": discussion_id}}
            ]
        }
        try:
            # Utiliser le paramètre 'query_filter' au lieu de 'filter'
            result, _ = qdrant_service.client.scroll(
                collection_name=self.collection_name,
                query_filter=filter_payload,
                limit=100  # Ajustez la limite selon vos besoins
            )
            history_list = []
            for point in result:
                history_list.append({"id": point.id, **point.payload})
            return history_list
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique : {e}")
            return []

    def search_similar_messages(self, query: str, top_k: int = 3) -> list:
        """
        Recherche des échanges similaires dans la collection 'history' de Qdrant.
        Calcule l'embedding de la requête et effectue une recherche par similarité.
        """
        query_vector = embedding_service.get_embedding(query)
        try:
            result = qdrant_service.search_similar(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            # On retourne directement la liste des résultats
            return result
        except Exception as e:
            logger.error(f"Erreur lors de la recherche des messages similaires : {e}")
            return []

    def delete_history(self, history_id: str) -> bool:
        """
        Supprime un échange de la collection 'history' dans Qdrant.
        """
        try:
            qdrant_service.delete_document(
                collection_name=self.collection_name,
                document_id=history_id
            )
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'historique '{history_id}' : {e}")
            return False


# Instance unique du service
history_service = HistoryService()
