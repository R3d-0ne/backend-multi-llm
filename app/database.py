import logging
from qdrant_client.http import models

from .services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)


def create_or_update_collections(collections_def: list) -> dict:
    """
    Crée ou met à jour les collections dans Qdrant à partir d'un tableau de définitions.

    Chaque élément de `collections_def` doit être un dictionnaire contenant :
      - "name" : le nom de la collection
      - "vector_size" : la dimension du vecteur
      - "distance" (optionnel, par défaut "Cosine") : le type de distance ("Cosine", "Euclidean", "Dot")

    Retourne un résumé des opérations effectuées.
    """
    summary = {}
    for coll_def in collections_def:
        name = coll_def.get("name")
        vector_size = coll_def.get("vector_size")
        distance = coll_def.get("distance", "Cosine")

        logger.info(f"Traitement de la collection '{name}' (vector_size={vector_size}, distance={distance})")
        try:
            # Récupérer la collection existante
            result = qdrant_service.client.get_collection(collection_name=name)
            # Accéder aux paramètres via les attributs
            if hasattr(result, "vectors"):
                existing_size = result.vectors.size
                existing_distance = result.vectors.distance
            else:
                existing_size = None
                existing_distance = None

            # Comparer la configuration existante avec la nouvelle
            if existing_size != vector_size or (existing_distance and existing_distance.lower() != distance.lower()):
                logger.info(
                    f"La collection '{name}' existe mais la configuration diffère (taille actuelle: {existing_size}, distance actuelle: {existing_distance}). Mise à jour...")
                qdrant_service.client.recreate_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=vector_size, distance=distance)
                )
                summary[name] = "Updated"
            else:
                logger.info(f"La collection '{name}' existe et est à jour.")
                summary[name] = "Up-to-date"
        except Exception as e:
            # Si la collection n'existe pas (vérifiez le message d'erreur ou un code spécifique)
            if "not found" in str(e).lower():
                logger.info(f"La collection '{name}' n'existe pas. Création en cours...")
                try:
                    qdrant_service.client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(size=vector_size, distance=distance)
                    )
                    summary[name] = "Created"
                except Exception as create_e:
                    logger.error(f"Erreur lors de la création de la collection '{name}': {create_e}")
                    summary[name] = f"Creation failed: {create_e}"
            else:
                logger.error(f"Erreur lors de la récupération de la collection '{name}': {e}")
                summary[name] = f"Error: {e}"
    return summary
