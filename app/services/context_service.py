from bson import ObjectId

from .mongo_service import mongo_service


class ContextService:
    def __init__(self):
        """ Initialise la collection 'contexts' """
        self.collection = mongo_service.get_collection("contexts")

    def create_context(self, name: str, content: str) -> str:
        """ Ajoute un contexte et retourne son ID """
        document = {"name": name, "content": content}
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def get_context(self, context_id: str) -> dict:
        """ Récupère un contexte par son ID """
        document = self.collection.find_one({"_id": ObjectId(context_id)})
        if document:
            document["_id"] = str(document["_id"])
        return document

    def get_all_contexts(self) -> list:
        """ Récupère tous les contextes stockés """
        return list(self.collection.find({}, {"_id": 0}))

    def delete_context(self, context_id: str) -> bool:
        """ Supprime un contexte par son ID """
        result = self.collection.delete_one({"_id": ObjectId(context_id)})
        return result.deleted_count > 0

# Instance unique
context_service = ContextService()
