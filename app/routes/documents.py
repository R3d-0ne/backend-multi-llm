from fastapi import APIRouter, HTTPException
from ..libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
from ..services.document_upload_service import DocumentUploadService

router = APIRouter()
upload_service = DocumentUploadService()


@router.post("/documents/", status_code=201)
async def create_document(source: str):
    """
    Dépose un document en utilisant le chemin source fourni et lance le pipeline de traitement.

    :param source: Chemin du fichier à déposer.
    :return: Un dictionnaire avec l'ID du document et le résultat du pipeline de traitement.
    """
    try:
        # Lancer le pipeline de traitement via l'orchestrateur.
        # L'orchestrateur se charge lui-même de déposer le document et de lancer toutes les étapes.
        orchestrator = TraitementOrchestrator()
        result = orchestrator.run_pipeline(source)
        return {
            "document_id": result.get("document_id"),
            "message": "Traitement lancé avec succès.",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors du dépôt : {e}")


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Récupère le chemin du document déposé à partir de son identifiant.

    :param document_id: Identifiant du document.
    :return: Un dictionnaire avec l'ID du document et son chemin.
    """
    try:
        path = upload_service.get_document_path(document_id)
        return {"document_id": document_id, "file_path": path}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document non trouvé : {e}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Supprime le document correspondant à l'identifiant fourni.

    :param document_id: Identifiant du document.
    :return: Un dictionnaire confirmant la suppression.
    """
    try:
        upload_service.delete_document(document_id)
        return {"document_id": document_id, "message": "Document supprimé avec succès."}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur lors de la suppression : {e}")
