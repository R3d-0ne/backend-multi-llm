import logging
from qdrant_client.http import models
from typing import Dict, Any

from .services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)


def create_or_update_collections(collections_def: list) -> dict:
    """
    Crée ou met à jour les collections dans Qdrant à partir d'un tableau de définitions.
    Seule la collection "documents" sera créée en mode hybride.

    Chaque élément de `collections_def` doit être un dictionnaire contenant :
      - "name" : le nom de la collection
      - "vector_size" : la dimension du vecteur
      - "distance" (optionnel, par défaut "Cosine") : le type de distance ("Cosine", "Euclidean", "Dot")

    Returns:
        Dictionnaire résumant les opérations effectuées
    """
    summary = {}
    for coll_def in collections_def:
        name = coll_def.get("name")
        vector_size = coll_def.get("vector_size")
        distance = coll_def.get("distance", "Cosine")
        
        # Vérifier si la collection existe
        collection_exists = qdrant_service.collection_exists(name)
        
        try:
            # Seule la collection "documents" doit être hybride
            if name == "documents":
                # Paramètres pour la collection hybride
                dense_size = vector_size
                entity_size = vector_size
                sparse_size = 10000  # Taille standard pour vecteurs épars
                
                logger.info(f"Traitement de la collection hybride '{name}' (dense={dense_size}, entity={entity_size}, sparse={sparse_size}, distance={distance})")
                
                if collection_exists:
                    logger.info(f"La collection '{name}' existe. Recréation en tant que collection hybride...")
                    qdrant_service.client.delete_collection(name)
                    
                # Créer la collection hybride
                qdrant_service.create_hybrid_collection(
                    collection_name=name,
                    dense_vector_size=dense_size,
                    entity_vector_size=entity_size,
                    sparse_vector_size=sparse_size,
                    distance=distance
                )
                summary[name] = collection_exists and "Updated (hybrid)" or "Created (hybrid)"
                logger.info(f"Collection hybride '{name}' créée avec succès.")
            else:
                # Collection standard pour les autres
                logger.info(f"Traitement de la collection standard '{name}' (vector_size={vector_size}, distance={distance})")
                
                if collection_exists:
                    # Vérifier si la configuration correspond
                    result = qdrant_service.client.get_collection(collection_name=name)
                    try:
                        existing_size = result.vectors.size
                        existing_distance = result.vectors.distance
                        
                        if existing_size != vector_size or (existing_distance and existing_distance.lower() != distance.lower()):
                            logger.info(f"La collection '{name}' existe mais la configuration diffère. Mise à jour...")
                            qdrant_service.client.recreate_collection(
                                collection_name=name,
                                vectors_config=models.VectorParams(size=vector_size, distance=distance)
                            )
                            summary[name] = "Updated"
                        else:
                            logger.info(f"La collection '{name}' existe et est à jour.")
                            summary[name] = "Up-to-date"
                    except AttributeError:
                        # Si la collection existe mais avec une structure différente
                        logger.info(f"La collection '{name}' existe mais a une structure différente. Recréation...")
                        qdrant_service.client.delete_collection(name)
                        qdrant_service.client.create_collection(
                            collection_name=name,
                            vectors_config=models.VectorParams(size=vector_size, distance=distance)
                        )
                        summary[name] = "Updated"
                else:
                    # Créer une nouvelle collection standard
                    logger.info(f"Création de la collection standard '{name}'...")
                    qdrant_service.client.create_collection(
                        collection_name=name,
                        vectors_config=models.VectorParams(size=vector_size, distance=distance)
                    )
                    summary[name] = "Created"
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la collection '{name}': {e}")
            summary[name] = f"Error: {e}"
    
    return summary
