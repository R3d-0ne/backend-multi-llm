from pydantic import BaseModel
from typing import Optional
import requests
import logging
import os
from fastapi import HTTPException
from dotenv import load_dotenv

from .context_service import context_service
from .discussions_service import discussions_service
from .message_service import message_service
from .settings_service import settings_service
from ..models.message import Message


class GenerateRequest(BaseModel):
    discussion_id: Optional[str] = None  # Optionnel : si non fourni, une discussion sera créée
    settings_id: Optional[str] = None      # Le setting_id à utiliser, s'il y en a plusieurs
    current_message: str
    additional_info: Optional[str] = None


logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Configuration du LLM
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")


class GenerateService:
    def __init__(self):
        # URL de l'API DeepSeek (Ollama) en local
        self.url = "http://host.docker.internal:11434/api/generate"

    def generate_response(
            self,
            discussion_id: Optional[str],
            settings_id: Optional[str],
            current_message: str,
            additional_info: Optional[str] = None
    ) -> dict:
        """
        Orchestre la génération d'une réponse.
        Si aucun discussion_id n'est fourni, il crée une nouvelle discussion.
        Utilise le settings_id si fourni pour récupérer les paramètres spécifiques.
        Ensuite, il construit le prompt via le service de contexte et envoie ce prompt à DeepSeek.
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

            # Appel du service de contexte pour construire et enregistrer le prompt final.
            # On passe les settings récupérés dans ce cas.
            context_result = context_service.save_full_context(
                discussion_id=discussion_id,
                current_message=current_message,
                additional_info=additional_info,
                settings_id=settings_id  # Corrected parameter name
            )
            context_id = context_result.get("id")
            # Récupérer le prompt stocké via le service de contexte.
            prompt_data = context_service.get_context(context_id)
            if not prompt_data or "prompt" not in prompt_data:
                raise HTTPException(status_code=500, detail="Erreur lors de la récupération du prompt")
            prompt = prompt_data["prompt"]
            logger.info(f"Prompt final construit :\n{prompt}")

            # Envoyer la requête à DeepSeek (Ollama)
            response = requests.post(
                self.url,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
            )
            if response.status_code != 200:
                logger.error(f"Erreur API DeepSeek: {response.text}")
                return {"error": f"Erreur avec DeepSeek: {response.text}"}
            data = response.json()
            answer = data.get("response", "")
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