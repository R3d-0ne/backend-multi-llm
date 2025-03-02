from fastapi import APIRouter, HTTPException

from ..models.context import Context
from ..services.context_service import context_service

router = APIRouter()


@router.post("/contexts", status_code=201)
async def create_context(context: Context):
    """
    Crée et enregistre un contexte (prompt final) qui regroupe toutes les informations (historique, settings, message actuel, etc.).
    """
    try:
        result = context_service.save_full_context(
            discussion_id=context.discussion_id,
            setting_id=context.setting_id,
            current_message=context.current_message,
            additional_info=context.additional_info  # Ce champ est optionnel
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/contexts/{context_id}")
async def get_context(context_id: str):
    """
    Récupère un contexte par son identifiant.
    """
    try:
        result = context_service.get_context(context_id)
        if not result:
            raise HTTPException(status_code=404, detail="Contexte non trouvé")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/contexts/{context_id}")
async def delete_context(context_id: str):
    """
    Supprime un contexte par son identifiant.
    """
    try:
        return context_service.delete_context(context_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
