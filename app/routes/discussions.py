from fastapi import APIRouter, HTTPException

from ..models.discussion import Discussion
from ..models.request_models import SearchDiscussionRequest, CreateDiscussionRequest
from ..services.discussions_service import discussions_service

router = APIRouter()
# Définition des routes FastAPI
@router.post("/discussions/", status_code=201)
async def create_discussion(request: CreateDiscussionRequest):
    # Créer une discussion avec le titre fourni ou par défaut
    discussion = Discussion(title=request.title)
    
    # Ajouter la discussion via le service
    result = discussions_service.add_discussion(discussion)
    
    return result


@router.get("/discussions/{discussion_id}")
async def read_discussion(discussion_id: str):
    return discussions_service.get_discussion(discussion_id)


@router.get("/discussions/")
async def read_all_discussions():
    return discussions_service.get_all_discussions()


@router.put("/discussions/{discussion_id}")
async def update_discussion(discussion_id: str, discussion: Discussion):
    return discussions_service.update_discussion(discussion_id, discussion)


@router.delete("/discussions/{discussion_id}")
async def delete_discussion(discussion_id: str):
    return discussions_service.delete_discussion(discussion_id)
