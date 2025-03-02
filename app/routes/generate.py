from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..services.generate_service import generate_service
from ..services.settings_service import settings_service

router = APIRouter()


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None


@router.post("/generate", status_code=200)
async def generate_response(request: GenerateRequest):
    """
    Orchestre la génération d'une réponse en construisant le prompt via le service de contexte
    et en l'envoyant à DeepSeek.
    """
    try:
        result = generate_service.generate_response(
            discussion_id=request.discussion_id,
            settings_id=request.settings_id,
            current_message=request.current_message,
            additional_info=request.additional_info
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
