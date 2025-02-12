from datetime import datetime
from .mongo_service import mongo_service


class HistoryService:
    def __init__(self):
        self.collection_name = "history"

    def add_history(self, discussion_id: str, question: str, response: str, context_id: str = None) -> str:
        """ Ajoute une interaction dans l'historique avec l'ID de la discussion """
        history_entry = {
            "discussion_id": discussion_id,  # ✅ Ajout de l'ID de la discussion
            "question": question,
            "response": response,
            "context_id": context_id,
            "timestamp": datetime.utcnow()
        }
        return mongo_service.insert_document(self.collection_name, history_entry)

    def get_all_history(self):
        """ Récupère tout l'historique des échanges """
        return mongo_service.get_all_documents(self.collection_name)

    def get_history_by_discussion(self, discussion_id: str):
        """ Récupère l'historique d'une discussion spécifique """
        return mongo_service.get_all_documents(self.collection_name, {"discussion_id": discussion_id})

    def get_history_by_id(self, history_id: str):
        """ Récupère un échange spécifique """
        return mongo_service.get_document_by_id(self.collection_name, history_id)

    def delete_history(self, history_id: str):
        """ Supprime un échange """
        return mongo_service.delete_document(self.collection_name, history_id)


# Singleton
history_service = HistoryService()
