import os
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId

# Charger les variables d'environnement
load_dotenv()


class MongoService:
    def __init__(self):
        """ Initialise la connexion à MongoDB """
        self.client = MongoClient(os.getenv("MONGO_URI", "mongodb://mongo:27017"))
        self.db = self.client[os.getenv("DB_NAME", "deepseek")]

    def get_collection(self, name):
        """ Récupère une collection MongoDB """
        return self.db[name]

    def insert_document(self, collection_name, document):
        """ Insère un document et retourne son ID """
        collection = self.get_collection(collection_name)
        result = collection.insert_one(document)
        return str(result.inserted_id)

    from bson import ObjectId

    def get_all_documents(self, collection_name, query={}):
        """ Récupère tous les documents d'une collection avec un filtre optionnel et convertit `_id` """
        collection = self.get_collection(collection_name)
        documents = list(collection.find(query))

        # 🔹 Convertir `_id` en string pour éviter les problèmes JSON
        for doc in documents:
            doc["_id"] = str(doc["_id"])

        return documents

    def get_document_by_id(self, collection_name, doc_id):
        """ Récupère un document par son ID """
        collection = self.get_collection(collection_name)
        document = collection.find_one({"_id": ObjectId(doc_id)})
        if document:
            document["_id"] = str(document["_id"])  # Convertir ObjectId en string
        return document

    def delete_document(self, collection_name, doc_id):
        """ Supprime un document par son ID """
        collection = self.get_collection(collection_name)
        result = collection.delete_one({"_id": ObjectId(doc_id)})
        return result.deleted_count > 0

    def update_document(self, collection_name, document_id, update_data):
        """Met à jour un document en fonction de son ID"""
        try:
            result = self.db[collection_name].update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
            return result.modified_count > 0  # ✅ Renvoie True si mise à jour réussie
        except Exception as e:
            print(f"❌ Erreur lors de la mise à jour du document {document_id} dans {collection_name}: {e}")
            return False

    # Singleton MongoDB


mongo_service = MongoService()
