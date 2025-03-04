import logging

from .OcrTask import OCRTask
from .PreprocessingTask import PreprocessingTask
from .TableDetectionTask import TableDetectionTask
from .UploadDocumentTask import UploadDocumentTask
from .EmbeddingTask import EmbeddingTask
from .StorageTask import StorageTask
from ..PipelineOrchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)


class TraitementOrchestrator:
    """
    Classe orchestrant l'ensemble des traitements.
    Chaque étape de traitement (classe héritant de Traitement) sera ajoutée à l'aide de la méthode add_step.
    La méthode run_pipeline exécute chaque étape dans l'ordre, en passant le résultat de l'étape précédente à la suivante.
    """

    def __init__(self):
        self.pipeline = PipelineOrchestrator()
        self.pipeline.add_step(UploadDocumentTask())
        self.pipeline.add_step(OCRTask())
        self.pipeline.add_step(TableDetectionTask())
        self.pipeline.add_step(PreprocessingTask())
        self.pipeline.add_step(EmbeddingTask())
        self.pipeline.add_step(StorageTask())

    def run_pipeline(self, initial_data: str) -> dict:
        """
        Exécute le pipeline de traitement en utilisant les données initiales fournies.

        :param initial_data: Donnée d'entrée pour démarrer le pipeline (ex: chemin du fichier à traiter).
        :return: Dictionnaire récapitulatif du résultat final du pipeline.
        """
        final_result = self.pipeline.run(initial_data)
        logger.info("Pipeline terminé, résultat final : %s", final_result)
        return final_result
