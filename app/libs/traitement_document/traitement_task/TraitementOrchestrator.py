import logging

from .OcrTask import OCRTask
from .PreprocessingTask import PreprocessingTask
from .TableDetectionTask import TableDetectionTask
from .UploadDocumentTask import UploadDocumentTask
from .LLMEnrichmentTask import LLMEnrichmentTask
from .EmbeddingTask import EmbeddingTask
from .StorageTask import StorageTask
from ..PipelineOrchestrator import PipelineOrchestrator
from ....services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)


class TraitementOrchestrator:
    """
    Classe orchestrant l'ensemble des traitements.
    Chaque étape de traitement (classe héritant de Traitement) sera ajoutée à l'aide de la méthode add_step.
    La méthode run_pipeline exécute chaque étape dans l'ordre, en passant le résultat de l'étape précédente à la suivante.
    """

    def __init__(self, use_hybrid_storage=True, use_llm_enrichment=True):
        self.pipeline = PipelineOrchestrator()
        self.pipeline.add_step(UploadDocumentTask())
        self.pipeline.add_step(OCRTask())
        self.pipeline.add_step(TableDetectionTask())
        self.pipeline.add_step(PreprocessingTask())
        
        # Ajout de l'étape d'enrichissement LLM (optionnel)
        if use_llm_enrichment:
            logger.info("TraitementOrchestrator - Enrichissement LLM activé pour le pipeline")
            self.pipeline.add_step(LLMEnrichmentTask())
        else:
            logger.info("TraitementOrchestrator - Enrichissement LLM désactivé pour le pipeline")
        
        self.pipeline.add_step(EmbeddingTask())
        
        # Configuration du stockage
        storage_task = StorageTask()
        storage_task.use_hybrid_storage = use_hybrid_storage
        self.pipeline.add_step(storage_task)
        
        # Initialisation de la collection si nécessaire
        self._initialize_collection(use_hybrid_storage)

    def _initialize_collection(self, use_hybrid_storage):
        """
        Initialise les collections nécessaires dans Qdrant.
        
        Args:
            use_hybrid_storage: Indique si l'on utilise le stockage hybride
        """
        try:
            collection_name = "documents"
            
            # Vérifier si la collection existe déjà
            if qdrant_service.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' déjà existante.")
                return
                
            # Créer la collection appropriée selon le mode de stockage
            if use_hybrid_storage:
                qdrant_service.create_hybrid_collection(
                    collection_name=collection_name,
                    dense_vector_size=768,
                    entity_vector_size=768,
                    sparse_vector_size=10000
                )
                logger.info(f"Collection hybride '{collection_name}' créée avec succès.")
            else:
                qdrant_service.create_collection(
                    collection_name=collection_name,
                    vector_size=768,
                    distance="Cosine"
                )
                logger.info(f"Collection standard '{collection_name}' créée avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la collection: {e}")
            # On continue même si l'initialisation échoue

    def run_pipeline(self, initial_data: str) -> dict:
        """
        Exécute le pipeline de traitement en utilisant les données initiales fournies.

        :param initial_data: Donnée d'entrée pour démarrer le pipeline (ex: chemin du fichier à traiter).
        :return: Dictionnaire récapitulatif du résultat final du pipeline.
        """
        logger.info("Démarrage du pipeline de traitement...")
        
        final_result = self.pipeline.run(initial_data)
        
        # Log uniquement les informations pertinentes, pas le contenu complet
        log_result = {
            "document_id": final_result.get("document_id"),
            "filename": final_result.get("filename"),
            "llm_enrichment_status": final_result.get("llm_enrichment_status", "non_utilisé"),
        }
        
        # Ajouter des infos LLM si disponibles
        if "llm_entities" in final_result:
            log_result["llm_entities_count"] = len(final_result.get("llm_entities", []))
        if "llm_keywords" in final_result:
            log_result["llm_keywords_count"] = len(final_result.get("llm_keywords", []))
        if "llm_document_type" in final_result:
            log_result["llm_document_type"] = final_result.get("llm_document_type")
        
        logger.info(f"Pipeline terminé, résultat: {log_result}")
        return final_result
