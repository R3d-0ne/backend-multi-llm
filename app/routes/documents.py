from fastapi import APIRouter, HTTPException, UploadFile, File
from ..libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
from ..services.document_upload_service import DocumentUploadService
import os
import tempfile

router = APIRouter()
upload_service = DocumentUploadService()


@router.post("/documents/", status_code=201)
async def create_document(file: UploadFile = File(...)):
    """
    Upload un document et lance le pipeline de traitement.
    
    :param file: Le fichier à uploader
    :return: Un dictionnaire avec l'ID du document et le résultat du pipeline de traitement
    """
    try:
        # Récupérer l'extension du fichier original
        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1].lower()

        # Créer un fichier temporaire avec l'extension appropriée
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Lancer le pipeline de traitement via l'orchestrateur
        orchestrator = TraitementOrchestrator()
        result = orchestrator.run_pipeline(temp_file_path)

        # Nettoyer le fichier temporaire
        os.unlink(temp_file_path)

        return {
            "success": True,
            "message": "Document uploadé et traitement lancé avec succès",
            "document_id": result.get("document_id"),
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de l'upload : {str(e)}")


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
