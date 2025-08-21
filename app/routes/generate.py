from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from ..services.service_compatibility import migration_manager
from ..services.settings_service import settings_service
from ..services.llm_service import llm_service

router = APIRouter()


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None  # Ajout du paramètre pour sélectionner le modèle


@router.post("/generate", status_code=200)
async def generate_response(request: GenerateRequest):
    """
    Orchestre la génération d'une réponse en construisant le prompt via le service de contexte
    et en l'envoyant au modèle LLM choisi par l'utilisateur.
    """
    try:
        # Utiliser le modèle spécifié pour cette requête si fourni
        generate_service = migration_manager.get_generate_service()
        result = generate_service.generate_response(
            discussion_id=request.discussion_id,
            settings_id=request.settings_id,
            current_message=request.current_message,
            additional_info=request.additional_info,
            model_id=request.model_id  # Passer le modèle choisi
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", status_code=200)
async def get_available_models() -> List[Dict[str, Any]]:
    """
    Renvoie la liste des modèles LLM disponibles pour l'interface utilisateur.
    """
    try:
        models = llm_service.get_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/select", status_code=200)
async def select_model(model_id: str):
    """
    Sélectionne le modèle LLM à utiliser par défaut.
    """
    try:
        success = llm_service.set_model(model_id)
        if success:
            return {"status": "success", "message": f"Modèle {model_id} sélectionné avec succès"}
        else:
            raise HTTPException(status_code=400, detail=f"Modèle {model_id} non disponible")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))