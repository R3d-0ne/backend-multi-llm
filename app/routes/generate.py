from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import des services originaux et de la couche de compatibilité
from ..services.generate_service import generate_service
from ..services.settings_service import settings_service
from ..services.llm_service import llm_service

try:
    from ..services.service_compatibility import generate_service_compat, migration_manager
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    COMPATIBILITY_AVAILABLE = False
    print("⚠️  Couche de compatibilité non disponible pour le service de génération")

router = APIRouter()


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None  # Ajout du paramètre pour sélectionner le modèle


def get_generate_service():
    """Retourne le service de génération approprié (refactorisé ou original)."""
    if COMPATIBILITY_AVAILABLE and hasattr(migration_manager, 'get_generate_service'):
        return migration_manager.get_generate_service()
    return generate_service


@router.post("/generate", status_code=200)
async def generate_response(request: GenerateRequest):
    """
    Orchestre la génération d'une réponse en construisant le prompt via le service de contexte
    et en l'envoyant au modèle LLM choisi par l'utilisateur.
    """
    try:
        # Utiliser le service approprié pour cette requête
        active_generate_service = get_generate_service()
        result = active_generate_service.generate_response(
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


@router.get("/services/migration-status", status_code=200)
async def get_migration_status():
    """
    Obtient le statut de migration des services refactorisés.
    """
    if not COMPATIBILITY_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Couche de compatibilité non disponible",
            "services": {}
        }
    
    try:
        status = migration_manager.get_migration_status()
        return {
            "status": "available",
            "message": "Statut de migration récupéré avec succès",
            "services": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération du statut: {str(e)}")


@router.post("/services/toggle-migration", status_code=200)
async def toggle_service_migration(service_name: str, use_refactored: bool = True):
    """
    Bascule un service vers sa version refactorisée ou originale.
    """
    if not COMPATIBILITY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Couche de compatibilité non disponible")
    
    try:
        valid_services = ['search', 'embedding', 'document', 'generate']
        if service_name not in valid_services:
            raise HTTPException(status_code=400, detail=f"Service invalide. Services valides: {valid_services}")
        
        migration_manager.migrate_service(service_name, use_refactored)
        
        return {
            "status": "success",
            "message": f"Service {service_name} basculé vers {'refactorisé' if use_refactored else 'original'}",
            "service": service_name,
            "using_refactored": use_refactored,
            "migration_status": migration_manager.get_migration_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la migration: {str(e)}")