from fastapi import APIRouter, HTTPException
from ..models.request_models import QuestionRequest
from ..services.generate_service import generate_service
from ..services.context_service import context_service
from ..services.history_service import history_service
from ..services.discussions_service import discussion_service  # 🔹 Gestion des discussions

router = APIRouter()

@router.post("/generate/")
async def generate_text(request: QuestionRequest):
    """ Génère une réponse à partir d'une question et d'un contexte, crée une discussion si nécessaire """

    # 🔹 Vérifier si un `context_id` est fourni
    if not request.context_id:
        raise HTTPException(status_code=400, detail="L'ID du contexte est requis")

    # 🔹 Vérifier que le contexte existe
    context = context_service.get_context(request.context_id)
    if not context:
        raise HTTPException(status_code=404, detail="Contexte non trouvé")

    # 🔹 Vérifier si `discussion_id` est fourni, sinon CRÉER UNE NOUVELLE DISCUSSION
    if not request.discussion_id:
        new_discussion = discussion_service.create_discussion(request.context_id)
        discussion_id = new_discussion  # Récupérer l'ID généré
        print(f"🔍 Nouvelle discussion créée: {discussion_id}")  # 🔍 Debug

    else:
        discussion_id = request.discussion_id  # Utiliser l'ID existant

    # 🔹 Récupérer le résumé de la discussion
    summary = discussion_service.get_summary(discussion_id)

    # 🔹 Générer la réponse en passant **les trois arguments requis**
    response = generate_service.generate_response(discussion_id, request.question, request.context_id)

    if "error" in response:
        raise HTTPException(status_code=500, detail=response["error"])

    # 🔹 Sauvegarder l'interaction dans l'historique
    history_service.add_history(discussion_id,request.question, response["response"], request.context_id)

    # 🔹 Mettre à jour le résumé de la discussion
    discussion_service.update_summary(discussion_id,request.question, response["response"])

    # 🔹 Retourner la réponse et l'ID de la discussion
    return {
        "discussion_id": discussion_id,
        "response": response["response"]
    }
