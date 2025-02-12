from fastapi import APIRouter, HTTPException

from ..services.history_service import history_service

router = APIRouter()


@router.post("/history/")
async def add_history(discussion_id: str, question: str, response: str, context_id: str = None):
    """ Ajoute une interaction à l'historique """
    history_id = history_service.add_history(question, response, context_id)
    return {"id": history_id, "discussion_id": discussion_id, "question": question, "response": response,
            "context_id": context_id}


@router.get("/history/")
async def list_history():
    """ Récupère tout l'historique des échanges """
    return history_service.get_all_history()


@router.get("/history/{history_id}")
async def get_history(history_id: str):
    """ Récupère un échange spécifique """
    history_entry = history_service.get_history_by_id(history_id)
    if not history_entry:
        raise HTTPException(status_code=404, detail="Échange introuvable")
    return history_entry


@router.delete("/history/{history_id}")
async def delete_history(history_id: str):
    """ Supprime un échange """
    success = history_service.delete_history(history_id)
    if not success:
        raise HTTPException(status_code=404, detail="Échange introuvable")
    return {"message": "Échange supprimé avec succès"}
