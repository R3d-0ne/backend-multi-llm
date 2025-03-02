from fastapi import APIRouter, HTTPException
from ..models.settings import Settings
from ..services.settings_service import settings_service

router = APIRouter()

@router.post("/settings", status_code=201)
def create_settings(setting: Settings):
    """
    Crée un nouveau document de settings.
    Le corps de la requête doit respecter le modèle Settings.
    """
    try:
        result = settings_service.create_settings(setting)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la création des settings")

@router.put("/settings/{setting_id}")
def update_settings(setting_id: str, setting: Settings):
    """
    Met à jour le document de settings identifié par setting_id.
    Le corps de la requête doit respecter le modèle Settings.
    """
    try:
        result = settings_service.update_settings(setting_id, setting)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour des settings")

@router.get("/settings/{setting_id}")
def get_settings_by_id(setting_id: str):
    """
    Récupère le document de settings correspondant à setting_id.
    """
    try:
        result = settings_service.get_settings_by_id(setting_id)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des settings")

@router.get("/settings")
def get_all_settings():
    """
    Récupère tous les documents de settings dans la collection.
    """
    try:
        result = settings_service.get_all_settings()
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de tous les settings")
