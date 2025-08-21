import os
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional, Union
from qdrant_client.http.models import Record, QueryResponse
import inspect
import json
from qdrant_client.models import NamedVector

from .config_service import config_service

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
                host=config_service.qdrant.host,
                port=config_service.qdrant.port,
                prefer_grpc=False
            )
            logger.info(f"Connecté à Qdrant ({config_service.qdrant.host}:{config_service.qdrant.port})")
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
        Supporte à la fois les collections standard et hybrides.
        
        Pour les collections hybrides, utilise le vecteur 'dense' par défaut.
        
        Retourne une liste de dictionnaires contenant l'ID, le score et le payload de chaque résultat.
        """
        try:
            # Conversion du vecteur de requête si nécessaire
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            if isinstance(query_vector, list) and query_vector and isinstance(query_vector[0], list):
                query_vector = query_vector[0]
            query_vector = [float(x) for x in query_vector]
            
            # Vérifier si la collection est hybride
            # Pour cela, nous essayons d'abord une recherche simple
            # Si ça échoue avec l'erreur "Collection requires specified vector name", alors c'est hybride
            is_hybrid = False
            try:
                # Essayer une recherche directe
                logger.info(f"Test de recherche simple sur {collection_name} (détection de type de collection)")
                self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=1
                )
                logger.info(f"Collection {collection_name} détectée comme standard")
            except Exception as e:
                error_message = str(e).lower()
                if "vector name" in error_message and "dense" in error_message:
                    is_hybrid = True
                    logger.info(f"Collection {collection_name} détectée comme hybride (contient les vecteurs nommés)")
                else:
                    logger.warning(f"Erreur lors du test de la collection: {e}")
            
            # Effectuer la recherche appropriée selon le type de collection
            if is_hybrid:
                logger.info(f"Utilisation de la recherche hybride (vecteur 'dense') pour {collection_name}")
                # Créer un NamedVector correctement formaté pour les collections hybrides
                
                named_vector = NamedVector(
                    name="dense",
                    vector=query_vector
                )
                
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_vector=named_vector,
                    limit=limit
                )
            else:
                # Collection standard, utiliser la recherche simple
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit
                )
            
            # Transformation des résultats
            formatted_results = []
            for res in search_result:
                result = {
                    "id": res.id,
                    "score": res.score,
                    **res.payload
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur recherche similaires: {e}")
            # Débogage amélioré
            if hasattr(e, "__dict__"):
                logger.debug(f"Détails de l'erreur: {e.__dict__}")
            return []

    def _debug_response_structure(self, obj, max_depth=2, current_depth=0) -> None:
        """
        Affiche la structure d'un objet de réponse pour faciliter le débogage.
        """
        if current_depth > max_depth:
            return
            
        if obj is None:
            logger.info("Objet de réponse: None")
            return
            
        try:
            # Structure de l'objet
            logger.info(f"Type d'objet: {type(obj).__name__}")
            
            # Attributs
            attrs = dir(obj)
            important_attrs = [attr for attr in attrs if not attr.startswith('_') and not callable(getattr(obj, attr))]
            logger.info(f"Attributs principaux: {', '.join(important_attrs) or 'aucun'}")
            
            # Pour les objets itérables
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                try:
                    first_items = list(obj)[:2]  # Prendre uniquement les 2 premiers éléments
                    if first_items:
                        logger.info(f"Premier élément de type: {type(first_items[0]).__name__}")
                        if current_depth < max_depth:
                            # Analyser récursivement le premier élément
                            self._debug_response_structure(first_items[0], max_depth, current_depth + 1)
                except Exception as e:
                    logger.warning(f"Impossible d'itérer sur l'objet: {e}")
            
            # Pour les objets dict-like
            if hasattr(obj, 'items') or isinstance(obj, dict):
                try:
                    keys = list(obj.keys())[:5] if hasattr(obj, 'keys') else []
                    logger.info(f"Clés principales: {', '.join(str(k) for k in keys) or 'aucune'}")
                except Exception as e:
                    logger.warning(f"Impossible d'accéder aux clés: {e}")
                    
            # Représentation en JSON si possible
            try:
                if hasattr(obj, '__dict__'):
                    logger.info(f"Aperçu JSON: {json.dumps(obj.__dict__, default=str)[:100]}...")
            except Exception:
                pass
                
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse de la structure: {e}")

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

    def create_hybrid_collection(
        self,
        collection_name: str,
        dense_vector_size: int = 768,
        entity_vector_size: int = 768,
        sparse_vector_size: int = 10000,
        distance: str = "Cosine"
    ) -> None:
        """
        Crée une collection hybride avec plusieurs espaces vectoriels : dense, entités et épars.
        
        Args:
            collection_name: Nom de la collection
            dense_vector_size: Taille du vecteur dense principal
            entity_vector_size: Taille du vecteur des entités
            sparse_vector_size: Taille du vecteur épars (tokens)
            distance: Métrique de distance (Cosine, Dot, Euclid)
        """
        try:
            # Suppression de la collection existante (si présente)
            if self.collection_exists(collection_name):
                logger.warning(f"Suppression collection hybride existante: {collection_name}")
                self.client.delete_collection(collection_name)

            # Conversion de la distance en énumération
            distance_enum = models.Distance[distance.upper()]
            
            # Définition des configurations de vecteurs
            vectors_config = {
                "dense": models.VectorParams(
                    size=dense_vector_size,
                    distance=distance_enum
                ),
                "entity": models.VectorParams(
                    size=entity_vector_size,
                    distance=distance_enum
                ),
                "sparse": models.VectorParams(
                    size=sparse_vector_size,
                    distance=models.Distance.DOT  # Dot product pour vecteurs épars
                )
            }

            # Création de la collection avec les configurations multiples
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config
            )
            logger.info(f"Collection hybride créée: {collection_name}")
        except KeyError:
            error_msg = f"Distance invalide: {distance}. Options: COSINE, EUCLID, DOT"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Échec création collection hybride: {e}")
            raise RuntimeError(f"Échec création collection hybride: {e}")

    def upsert_hybrid_document(
        self,
        collection_name: str,
        document_id: str,
        vectors: Dict[str, List[float]],
        payload: Dict[str, Any]
    ) -> None:
        """
        Insère ou met à jour un document dans une collection hybride avec plusieurs vecteurs.
        
        Args:
            collection_name: Nom de la collection
            document_id: ID du document
            vectors: Dictionnaire des vecteurs (ex: {"dense": [...], "entity": [...], "sparse": [...]})
            payload: Métadonnées du document
        """
        try:
            # Conversion et normalisation des vecteurs
            normalized_vectors = {}
            for vec_name, vector in vectors.items():
                # Conversion du vecteur s'il s'agit par exemple d'un numpy.ndarray
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                # Si le vecteur est une liste de listes, on sélectionne la première
                if isinstance(vector, list) and vector and isinstance(vector[0], list):
                    vector = vector[0]
                # Conversion explicite en float
                normalized_vectors[vec_name] = [float(x) for x in vector]

            point = models.PointStruct(
                id=document_id,
                vector=normalized_vectors,
                payload=payload
            )

            self.client.upsert(
                collection_name=collection_name,
                wait=True,
                points=[point]
            )
            logger.info(f"Document hybride upserté: {document_id}")
        except Exception as e:
            logger.error(f"Échec upsert document hybride {document_id}: {e}")
            raise

    def hybrid_search(
        self,
        collection_name: str,
        vectors: Dict[str, Union[List[float], Dict[str, Any]]],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Effectue une recherche hybride en utilisant plusieurs vecteurs avec pondération.
        
        Args:
            collection_name: Nom de la collection
            vectors: Dictionnaire des vecteurs de requête, avec poids optionnels
                    Format: {"dense": vector_list} ou 
                            {"dense": {"vector": vector_list, "weight": 0.7}}
            limit: Nombre maximum de résultats
            filter_conditions: Conditions de filtrage (optionnel)
            
        Returns:
            Liste des résultats de recherche (points)
        """
        try:
            from qdrant_client.models import NamedVector

            # Normalisation des vecteurs et préparation des NamedVectors
            named_vectors = []
            weights = {}
            
            for vec_name, value in vectors.items():
                # Extraction et normalisation du vecteur
                if isinstance(value, dict) and "vector" in value:
                    # Format avec poids
                    vector = value["vector"]
                    weight = value.get("weight", 1.0)
                else:
                    # Format simple (vecteur seul)
                    vector = value
                    weight = 1.0
                
                # Normalisation du vecteur
                if hasattr(vector, "tolist"):
                    vector = vector.tolist()
                if isinstance(vector, list) and vector and isinstance(vector[0], list):
                    vector = vector[0]
                vector = [float(x) for x in vector]
                
                # Création du NamedVector
                named_vector = NamedVector(
                    name=vec_name,
                    vector=vector
                )
                
                named_vectors.append(named_vector)
                weights[vec_name] = weight
            
            # Création du filtre si nécessaire
            search_filter = None
            if filter_conditions:
                search_filter = self._build_filter(filter_conditions)
            
            # Cas simple: un seul vecteur sans poids particulier
            if len(named_vectors) == 1 and weights[named_vectors[0].name] == 1.0:
                logger.info(f"Recherche simple avec vecteur nommé '{named_vectors[0].name}' sur {collection_name}")
                response = self.client.search(
                    collection_name=collection_name,
                    query_vector=named_vectors[0],
                    limit=limit,
                    query_filter=search_filter,
                    with_payload=True
                )
            else:
                # Approche 1: Trouver le vecteur principal (généralement 'dense') et l'utiliser seul
                # Cette approche est la plus simple et fonctionne bien dans la plupart des cas
                main_vector_name = "dense" if "dense" in weights else list(weights.keys())[0]
                main_vector_weight = weights[main_vector_name]
                main_vector = next((v for v in named_vectors if v.name == main_vector_name), named_vectors[0])
                
                logger.info(f"Recherche hybride simplifiée: utilisation du vecteur principal '{main_vector_name}' (poids {main_vector_weight:.1f})")
                
                response = self.client.search(
                    collection_name=collection_name,
                    query_vector=main_vector,
                    limit=limit,
                    query_filter=search_filter,
                    with_payload=True
                )
                
                # Ajouter un log pour mentionner les autres vecteurs qui n'ont pas été utilisés
                if len(named_vectors) > 1:
                    other_vectors = [v.name for v in named_vectors if v.name != main_vector_name]
                    logger.info(f"Note: les vecteurs secondaires {other_vectors} n'ont pas été utilisés dans cette recherche")
            
            # Transformation de la réponse
            results = []
            for point in response:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    **point.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche hybride: {e}")
            # Débogage amélioré
            if hasattr(e, "__dict__"):
                logger.debug(f"Détails de l'erreur: {e.__dict__}")
            return []

    def _build_filter(self, conditions: Dict[str, Any]) -> models.Filter:
        """
        Construit un filtre Qdrant à partir d'un dictionnaire de conditions.
        
        Args:
            conditions: Dictionnaire de conditions de filtrage
            
        Returns:
            Filtre Qdrant
        """
        must_conditions = []
        
        for key, value in conditions.items():
            if key == "upload_date_range" and isinstance(value, dict):
                if "start" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key="upload_date",
                            range=models.Range(
                                gte=value["start"]
                            )
                        )
                    )
                if "end" in value:
                    must_conditions.append(
                        models.FieldCondition(
                            key="upload_date",
                            range=models.Range(
                                lte=value["end"]
                            )
                        )
                    )
            elif key.startswith("has_") and isinstance(value, bool) and value:
                list_name = key[4:]  # Retirer le préfixe "has_"
                must_conditions.append(
                    models.FieldCondition(
                        key=list_name,
                        match=models.MatchAny(any=True)
                    )
                )
            elif isinstance(value, (str, int, float, bool)):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=must_conditions) if must_conditions else None


# Instance singleton du service
qdrant_service = QdrantService()
