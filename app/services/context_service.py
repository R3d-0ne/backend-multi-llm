import uuid
import logging
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

from .embedding_service import embedding_service
from .message_service import message_service
from .qdrant_service import qdrant_service
from .settings_service import settings_service

logger = logging.getLogger(__name__)


class ContextService:
    def __init__(self):
        self.collection_name = "contexts"
        # On suppose que la collection "contexts" a déjà été créée dans Qdrant.
        logger.info(f"Utilisation de la collection '{self.collection_name}' pour les contextes.")

    def build_prompt(
            self,
            discussion_id: str,
            setting_id: str,
            current_message: str,
            history: list[str],
            settings: dict[str, any],
            additional_info: str | None = None
    ) -> str:
        # Partie "System" : on regroupe les informations système.
        system_parts = []
        if settings and "content" in settings:
            system_parts.append(settings["content"])
        if additional_info:
            system_parts.append(additional_info)
        system_message = "\n".join(system_parts)

        # Construction de la conversation en assignant les rôles.
        # On suppose que l'historique est une liste de messages en alternance
        # (commençant par l'utilisateur).
        conversation = []
        for i, msg in enumerate(history):
            role = "user" if i % 2 == 0 else "assistant"
            conversation.append({"role": role, "content": msg})
        # Le message actuel est considéré comme venant de l'utilisateur.
        conversation.append({"role": "user", "content": current_message})

        # Formatage des messages selon le template DeepSeek.
        formatted_messages = []
        num_messages = len(conversation)
        for i, msg in enumerate(conversation):
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_messages.append(f"<｜User｜>{content}")
            elif role == "assistant":
                # Si ce n'est pas le dernier message, on ajoute la balise de fin.
                if i != num_messages - 1:
                    formatted_messages.append(f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>")
                else:
                    formatted_messages.append(f"<｜Assistant｜>{content}")

        # Si le dernier message n'est pas de l'assistant, on ajoute la balise finale.
        if conversation and conversation[-1]["role"] != "assistant":
            formatted_messages.append("<｜Assistant｜>")

        # On assemble le tout en séparant la partie système et la conversation par deux retours à la ligne.
        parts = []
        if system_message:
            parts.append(system_message)
        parts.extend(formatted_messages)

        return "\n\n".join(parts)

    def save_full_context(
            self,
            discussion_id: str,
            current_message: str,
            settings_id: str,
            additional_info: Optional[str] = None
    ) -> dict:
        """
        Construit le prompt final à partir de l'ID de la discussion, du message actuel,
        et récupère automatiquement l'historique (les 10 derniers messages) via le service messages
        ainsi que les settings via leur ID (settings_id).
        Calcule l'embedding du prompt final et l'enregistre dans Qdrant.
        """
        try:
            # Récupérer l'historique via le service messages
            messages = message_service.get_messages_by_discussion(discussion_id)

            # Conserver uniquement les 10 derniers messages
            if messages and len(messages) > 10:
                messages = messages[-10:]
            history_texts = [
                f"{record.get('sender', 'inconnu')}: {record.get('text', '')}"
                for record in messages
            ]

            # Récupérer les settings par leur ID
            settings_data = settings_service.get_settings_by_id(settings_id)
            if not settings_data:
                raise HTTPException(status_code=404, detail=f"Settings non trouvés pour l'ID {settings_id}")

            # Construire le prompt final
            prompt = self.build_prompt(discussion_id, settings_id, current_message, history_texts, settings_data,
                                       additional_info)
            logger.info(f"Prompt construit:\n{prompt}")

            # Calculer l'embedding du prompt
            vector = embedding_service.get_embedding(prompt)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            if isinstance(vector, list) and vector and isinstance(vector[0], list):
                vector = vector[0]

            context_id = str(uuid.uuid4())
            payload = {
                "discussion_id": discussion_id,
                "prompt": prompt,
                "history": history_texts,
                "settings": settings_data,
                "additional_info": additional_info or "",
                "current_message": current_message
            }
            qdrant_service.upsert_document(
                collection_name=self.collection_name,
                document_id=context_id,
                vector=vector,
                payload=payload
            )
            return {"message": "Contexte enregistré", "id": context_id}
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du contexte: {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'enregistrement du contexte")

    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère un contexte par son ID dans la collection 'contexts'.
        """
        try:
            result = qdrant_service.get_document(
                collection_name=self.collection_name,
                document_id=context_id
            )
            if result and isinstance(result, list) and len(result) > 0:
                return result[0].payload
            return None
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du contexte {context_id} : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la récupération du contexte")

    def delete_context(self, context_id: str) -> dict:
        """
        Supprime un contexte de la collection 'contexts'.
        """
        try:
            qdrant_service.delete_document(
                collection_name=self.collection_name,
                document_id=context_id
            )
            return {"message": "Contexte supprimé", "id": context_id}
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du contexte {context_id} : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de la suppression du contexte")


# Instance singleton pour une utilisation centralisée
context_service = ContextService()
