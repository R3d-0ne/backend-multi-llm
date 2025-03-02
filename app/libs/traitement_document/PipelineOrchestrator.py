
from typing import List, Any

from .traitement_task.UploadDocumentTask import UploadDocumentTask


class PipelineOrchestrator:
    """
    Classe orchestrant l'ensemble des traitements.
    Chaque étape de traitement (classe héritant de Traitement) sera ajoutée à l'aide de la méthode add_step.
    La méthode run exécute chaque étape dans l'ordre, en passant le résultat de l'étape précédente à la suivante.
    """
    def __init__(self):
        self.steps: List[Any] = []

    def add_step(self, step: Any) -> None:
        """ Ajoute une étape de traitement à la chaîne """
        self.steps.append(step)

    def run(self, data: Any) -> Any:
        """ Exécute chaque étape de traitement dans l'ordre """
        result = data
        for step in self.steps:
            result = step.run(result)
        return result






