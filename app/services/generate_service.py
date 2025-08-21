from pydantic import BaseModel
from typing import Optional
import requests
import logging
from fastapi import HTTPException

from .context_service import context_service
from .discussions_service import discussions_service
from .message_service import message_service
from .settings_service import settings_service
from .llm_service import llm_service  # Importation du service LLM
from ..models.message import Message


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None  # Optionnel : si non fourni, une discussion sera créée
    settings_id: Optional[str] = None  # Le setting_id à utiliser, s'il y en a plusieurs
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None  # Ajout du paramètre pour sélectionner le modèle


logger = logging.getLogger(__name__)


class GenerateService:
    def __init__(self):
        pass  # Plus besoin d'initialiser l'URL ici car on utilise llm_service

    def generate_response(
            self,
            discussion_id: Optional[str],
            settings_id: Optional[str],
            current_message: str,
            additional_info: Optional[str] = None,
            model_id: Optional[str] = None  # Nouveau paramètre pour le modèle
    ) -> dict:
        """
        Orchestre la génération d'une réponse.
        Si aucun discussion_id n'est fourni, il crée une nouvelle discussion.
        Utilise le settings_id si fourni pour récupérer les paramètres spécifiques.
        Utilise le model_id si fourni pour sélectionner le modèle LLM.
        Ensuite, il construit le prompt via le service de contexte et envoie ce prompt au LLM.
        """
        try:
            # Si discussion_id n'est pas fourni, créer une nouvelle discussion via le service de discussions.
            if not discussion_id:
                new_discussion = discussions_service.add_discussion("Nouvelle discussion")
                discussion_id = new_discussion.get("id")
                logger.info(f"Nouvelle discussion créée avec l'ID: {discussion_id}")

            # Récupérer les settings à partir du settings_id s'il est fourni, sinon utiliser le settings par défaut.
            if settings_id:
                settings_data = settings_service.get_settings_by_id(settings_id)
                if not settings_data:
                    logger.error(f"Settings non trouvés pour l'ID {settings_id}")
                    return {"message": "Settings non trouvés", "id": None}
            else:
                settings_data = settings_service.get_settings()

            # Extraire les paramètres du modèle depuis les settings s'ils existent
            temperature = 0.1  # Valeur par défaut
            max_tokens = 1024  # Valeur par défaut
            system_prompt = None  # Valeur par défaut

            # Récupérer les paramètres depuis settings_data si disponibles
            if settings_data and "payload" in settings_data:
                payload = settings_data.get("payload", {})
                if isinstance(payload, dict):
                    temperature = payload.get("temperature", temperature)
                    max_tokens = payload.get("max_tokens", max_tokens)
                    system_prompt = payload.get("system_prompt", system_prompt)

            # Appel du service de contexte pour construire et enregistrer le prompt final.
            context_result = context_service.save_full_context(
                discussion_id=discussion_id,
                current_message=current_message,
                additional_info=additional_info,
                settings_id=settings_id
            )
            context_id = context_result.get("id")

            # Récupérer le prompt stocké via le service de contexte.
            prompt_data = context_service.get_context(context_id)
            if not prompt_data or "prompt" not in prompt_data:
                raise HTTPException(status_code=500, detail="Erreur lors de la récupération du prompt")
            prompt = prompt_data["prompt"]
            logger.info(f"Prompt final construit :\n{prompt}")

            # Utiliser le service LLM pour générer une réponse
            # en lui passant le modèle spécifié si fourni
            answer = llm_service.generate_response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                model_override=model_id
            )

            # Enregistrer le message de l'utilisateur
            user_message = Message(
                discussion_id=discussion_id,
                sender="user",
                text=current_message
            )
            message_service.send_message(user_message)

            # Enregistrer la réponse de l'assistant
            assistant_message = Message(
                discussion_id=discussion_id,
                sender="assistant",
                text=answer
            )
            message_service.send_message(assistant_message)

            return {"response": answer, "context_id": context_id, "discussion_id": discussion_id}
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la génération de la réponse")


# Instance singleton
generate_service = GenerateService()