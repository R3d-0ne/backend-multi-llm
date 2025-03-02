from fastapi import APIRouter, HTTPException
from ..models.request_models import AddHistoryRequest, SearchHistoryRequest
from ..services.history_service import history_service

router = APIRouter()


@router.post("/history/")
async def add_history(request: AddHistoryRequest):
    """ Ajoute un échange à l'historique d'une discussion """

    if not request.discussion_id or not request.question or not request.response:
        raise HTTPException(status_code=400, detail="discussion_id, question et response sont requis")

    history_id = history_service.add_history(request.discussion_id, request.question, request.response)

    return {"history_id": history_id}


@router.get("/history/{discussion_id}")
async def get_history(discussion_id: str):
    """ Récupère l'historique d'une discussion spécifique """

    history = history_service.get_history_by_discussion(discussion_id)
    if not history:
        raise HTTPException(status_code=404, detail="Aucun historique trouvé pour cette discussion")

    return {"history": history}


@router.post("/history/search/")
async def search_history(request: SearchHistoryRequest):
    """ Recherche des échanges similaires dans l'historique """

    if not request.query:
        raise HTTPException(status_code=400, detail="La requête de recherche est requise")

    results = history_service.search_similar_messages(request.query)
    return {"results": results}


@router.delete("/history/{history_id}")
async def delete_history(history_id: str):
    """ Supprime un échange de l'historique """

    await history_service.delete_history(history_id)
    return {"message": f"Historique {history_id} supprimé avec succès"}
