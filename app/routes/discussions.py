from fastapi import APIRouter, HTTPException
from ..models.request_models import DiscussionRequest, DiscussionRequest
from ..services.discussions_service import discussion_service

router = APIRouter()


@router.get("/discussions")
async def get_all_discussions():
    """ Récupère la liste de toutes les discussions """
    discussions = discussion_service.get_all_discussions()
    return {"discussions": discussions}


@router.post("/discussions")
async def create_discussion(request: DiscussionRequest):
    """ Crée une nouvelle discussion """
    discussion_id = discussion_service.create_discussion(request.context_id)
    return {"status": "success", "discussion_id": discussion_id}


@router.get("/discussions/{discussion_id}")
async def get_discussion(discussion_id: str):
    """ Récupère une discussion spécifique """
    discussion = discussion_service.get_discussion(discussion_id)
    if not discussion:
        raise HTTPException(status_code=404, detail="Discussion non trouvée")
    return {"discussion": discussion}


@router.get("/discussions/{discussion_id}/summary")
async def get_summary(discussion_id: str):
    """ Récupère le résumé d'une discussion """
    summary = discussion_service.get_summary(discussion_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Discussion non trouvée")
    return {"discussion_id": discussion_id, "summary": summary}


@router.patch("/discussions/{discussion_id}/summary")
async def update_summary(discussion_id: str, request: DiscussionRequest):
    """ Met à jour le résumé d'une discussion avec une nouvelle entrée """
    discussion_service.update_summary(discussion_id, request.question, request.response)
    return {"status": "success", "message": "Résumé mis à jour"}


@router.delete("/discussions/{discussion_id}")
async def delete_discussion(discussion_id: str):
    """ Supprime une discussion par son ID """
    success = discussion_service.delete_discussion(discussion_id)
    if not success:
        raise HTTPException(status_code=404, detail="Discussion non trouvée")
    return {"status": "success", "message": f"Discussion {discussion_id} supprimée"}
