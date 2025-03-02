from fastapi import APIRouter, HTTPException, Query
from ..models.message import Message  # Votre modèle Message
from ..services.message_service import message_service

router = APIRouter()

@router.post("/messages/", status_code=201)
async def create_message(message: Message):
    """
    Crée et enregistre un nouveau message.
    Chaque message est inséré en tant que nouveau document.
    """
    try:
        return message_service.send_message(message)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de l'envoi du message")

@router.get("/messages/")
async def get_messages(
    discussion_id: str = Query(..., description="ID de la discussion pour laquelle récupérer les messages")
):
    """
    Récupère tous les messages associés à une discussion donnée.
    """
    try:
        return message_service.get_messages_by_discussion(discussion_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des messages")

@router.put("/messages/{message_id}")
async def update_message(message_id: str, new_metadata: dict):
    """
    Met à jour les métadonnées d'un message existant.
    """
    try:
        return message_service.update_message(message_id, new_metadata)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la mise à jour du message")

@router.delete("/messages/{message_id}")
async def delete_message(message_id: str):
    """
    Supprime un message identifié par son ID.
    """
    try:
        return message_service.delete_message(message_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur lors de la suppression du message")
