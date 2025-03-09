from fastapi import APIRouter, HTTPException, UploadFile, File, Body
from ..libs.traitement_document.traitement_task.TraitementOrchestrator import TraitementOrchestrator
from ..services.document_upload_service import DocumentUploadService
from ..services.document_service import DocumentService
from typing import Dict, Any
import os
import tempfile
from datetime import datetime

router = APIRouter()
upload_service = DocumentUploadService()
document_service = DocumentService()


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

        # Préparer les données initiales pour le pipeline
        initial_data = {
            'temp_file_path': temp_file_path,
            'filename': original_filename,
            'upload_date': datetime.now().isoformat()
        }

        # Lancer le pipeline de traitement via l'orchestrateur
        orchestrator = TraitementOrchestrator()
        result = orchestrator.run_pipeline(initial_data)

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
    Récupère les informations d'un document à partir de son identifiant.

    :param document_id: Identifiant du document.
    :return: Un dictionnaire avec les informations du document.
    """
    try:
        document = document_service.get_document(document_id)
        return document
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document non trouvé : {e}")


@router.get("/documents/")
async def list_documents():
    """
    Liste tous les documents disponibles.

    :return: Une liste de documents avec leurs métadonnées.
    """
    try:
        documents = document_service.list_documents()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du listage des documents : {e}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Supprime le document correspondant à l'identifiant fourni.

    :param document_id: Identifiant du document.
    :return: Un dictionnaire confirmant la suppression.
    """
    try:
        document_service.delete_document(document_id)
        return {"document_id": document_id, "message": "Document supprimé avec succès."}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur lors de la suppression : {e}")


@router.patch("/documents/{document_id}")
async def update_document_metadata(document_id: str, metadata: Dict[str, Any] = Body(...)):
    """
    Met à jour les métadonnées d'un document.

    :param document_id: Identifiant du document.
    :param metadata: Nouvelles métadonnées à associer au document.
    :return: Le document mis à jour avec ses métadonnées.
    """
    try:
        updated_document = document_service.update_document_metadata(document_id, metadata)
        return updated_document
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur lors de la mise à jour : {e}")
