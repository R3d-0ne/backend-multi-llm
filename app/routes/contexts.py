from fastapi import APIRouter, HTTPException

from ..models.request_models import ContextRequest
from ..services.context_service import context_service

router = APIRouter()


@router.post("/contexts/")
async def create_context(request: ContextRequest):
    """ Ajoute un nouveau contexte """
    context_id = context_service.create_context(request.name, request.content)
    return {"id": context_id, "name": request.name, "content": request.content}


@router.get("/contexts/")
async def list_contexts():
    """ Récupère tous les contextes """
    return context_service.get_all_contexts()


@router.get("/contexts/{context_id}")
async def get_context(context_id: str):
    """ Récupère un contexte spécifique """
    context = context_service.get_context(context_id)
    if not context:
        raise HTTPException(status_code=404, detail="Contexte introuvable")
    return context


@router.delete("/contexts/{context_id}")
async def delete_context(context_id: str):
    """ Supprime un contexte """
    success = context_service.delete_context(context_id)
    if not success:
        raise HTTPException(status_code=404, detail="Contexte introuvable")
    return {"message": "Contexte supprimé avec succès"}
