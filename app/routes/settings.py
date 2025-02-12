from fastapi import APIRouter

from ..services.settings_service import settings_service

router = APIRouter()


@router.put("/settings/")
async def update_settings(settings: dict):
    """ Met à jour les paramètres globaux """
    return settings_service.update_settings(settings)


@router.get("/settings/")
async def get_settings():
    """ Récupère les paramètres actuels """
    return settings_service.get_settings()
