from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ..services.generate_service import generate_service
from ..services.settings_service import settings_service
from ..services.llm_service import llm_service

router = APIRouter()


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None


@router.post("/generate", status_code=200)
async def generate_response(request: GenerateRequest):
    """
    Orchestre la génération d'une réponse en construisant le prompt via le service de contexte
    et en l'envoyant au modèle LLM choisi par l'utilisateur.
    """
    try:
        result = generate_service.generate_response(
            discussion_id=request.discussion_id,
            settings_id=request.settings_id,
            current_message=request.current_message,
            additional_info=request.additional_info,
            model_id=request.model_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_available_models():
    """Liste les modèles LLM disponibles"""
    try:
        return llm_service.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/select")
async def select_model(model_id: str = Query(..., description="ID du modèle à sélectionner")):
    """Sélectionne un modèle LLM par défaut"""
    try:
        success = llm_service.set_model(model_id)
        if success:
            return {"message": f"Modèle {model_id} sélectionné avec succès", "model_id": model_id}
        else:
            raise HTTPException(status_code=400, detail=f"Modèle {model_id} non disponible")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))