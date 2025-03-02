import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import Record, QueryResponse

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Configuration Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))


class QdrantService:
    _instance = None
    client: QdrantClient = None

    def __new__(cls):
        """Implémentation du pattern Singleton"""
        if cls._instance is None:
            cls._instance = super(QdrantService, cls).__new__(cls)
            try:
                cls._instance._initialize()
            except Exception as e:
                logger.error(f"Erreur d'initialisation: {e}")
                cls._instance = None
                raise
        return cls._instance

    def _initialize(self):
        """Initialise la connexion à Qdrant"""
        try:
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
                prefer_grpc=False
            )
            logger.info(f"Connecté à Qdrant ({QDRANT_HOST}:{QDRANT_PORT})")
        except Exception as e:
            logger.error(f"Échec de connexion: {e}")
            raise RuntimeError("Connexion Qdrant impossible")

    def collection_exists(self, collection_name: str) -> bool:
        """Vérifie l'existence d'une collection."""
        try:
            return self.client.collection_exists(collection_name)
        except Exception as e:
            logger.error(f"Erreur vérification collection: {e}")
            return False

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine"
    ) -> None:
        """
        Crée ou remplace une collection.
        Le vecteur utilisé pour les documents devra être de dimension `vector_size`.
        """
        try:
            # Suppression de la collection existante (si présente)
            if self.collection_exists(collection_name):
                logger.warning(f"Suppression collection existante: {collection_name}")
                self.client.delete_collection(collection_name)

            # Conversion de la distance en énumération
            distance_enum = models.Distance[distance.upper()]

            # Création de la collection avec la configuration des vecteurs
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_enum
                )
            )
            logger.info(f"Collection créée: {collection_name}")
        except KeyError:
            error_msg = f"Distance invalide: {distance}. Options: COSINE, EUCLID, DOT"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Échec création collection: {e}")
            raise RuntimeError(f"Échec création collection: {e}")

    def upsert_document(
        self,
        collection_name: str,
        document_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> None:
        """
        Insère ou met à jour un document dans la collection.
        S'assure que le vecteur est une liste plate de nombres (floats).
        """
        try:
            # Conversion du vecteur s'il s'agit par exemple d'un numpy.ndarray
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            # Si le vecteur est une liste de listes, on sélectionne la première
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]
            # Conversion explicite en float
            vector = [float(x) for x in vector]

            point = models.PointStruct(
                id=document_id,
                vector=vector,
                payload=payload
            )

            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[point]
            )
            logger.info(f"Document upserté: {document_id}")
        except Exception as e:
            logger.error(f"Échec upsert document {document_id}: {e}")
            raise

    def get_document(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[List[Record]]:
        """
        Récupère un document par son ID et renvoie toutes ses données (ID, vecteur et payload)
        sous forme d'une liste de Record.
        """
        try:
            response = self.client.retrieve(
                collection_name=collection_name,
                ids=[document_id],
                with_vectors=True,
                with_payload=True
            )
            if not response:
                return None
            return response
        except Exception as e:
            logger.error(f"Erreur récupération document {document_id}: {e}")
            return None

    from typing import List, Dict, Any

    def search_similar(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Recherche des documents similaires à partir d'un vecteur de requête.
        Retourne une liste de dictionnaires contenant l'ID, le score et le payload de chaque résultat.
        """
        try:
            # Conversion du vecteur de requête si nécessaire
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            if isinstance(query_vector, list) and query_vector and isinstance(query_vector[0], list):
                query_vector = query_vector[0]
            query_vector = [float(x) for x in query_vector]

            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )

            # Transformation de la réponse en liste de dictionnaires
            results = []
            for point in response.result:
                results.append({
                    "id": point.id,
                    "score": point.score if hasattr(point, "score") else None,
                    "payload": point.payload
                })
            return results
        except Exception as e:
            logger.error(f"Erreur recherche similaires: {e}")
            raise

    def delete_document(
        self,
        collection_name: str,
        document_id: str
    ) -> None:
        """
        Supprime un document identifié par son ID dans la collection.
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=[document_id])
            )
            logger.info(f"Document supprimé: {document_id}")
        except Exception as e:
            logger.error(f"Échec suppression document {document_id}: {e}")
            raise

    def update_document_metadata(
        self,
        collection_name: str,
        document_id: str,
        new_metadata: Dict[str, Any]
    ) -> None:
        """
        Met à jour les métadonnées (payload) d'un document.
        """
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=new_metadata,
                points=[document_id]
            )
            logger.info(f"Métadonnées mises à jour: {document_id}")
        except Exception as e:
            logger.error(f"Échec mise à jour métadonnées {document_id}: {e}")
            raise


# Instance singleton du service
qdrant_service = QdrantService()
