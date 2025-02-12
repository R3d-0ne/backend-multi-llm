from .mongo_service import mongo_service


class SettingsService:
    def __init__(self):
        self.collection_name = "settings"

    def update_settings(self, settings: dict):
        """ Met à jour les paramètres (écrase les anciens) """
        mongo_service.get_collection(self.collection_name).update_one({}, {"$set": settings}, upsert=True)
        return {"message": "Paramètres mis à jour"}

    def get_settings(self):
        """ Récupère les paramètres actuels """
        settings = mongo_service.get_collection(self.collection_name).find_one({}, {"_id": 0})
        return settings if settings else {}


# Singleton
settings_service = SettingsService()
