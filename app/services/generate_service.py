import requests
import os
from .discussions_service import discussion_service
from .history_service import history_service
from .context_service import context_service

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")


class GenerateService:
    def __init__(self):
        self.url = f"{OLLAMA_URL}/api/generate"

    def format_messages(self, history):
        """Transforme l'historique des messages pour l'intégrer au prompt"""
        formatted_messages = []
        for msg in history:
            if "question" in msg and "response" in msg:
                formatted_messages.append(f"Utilisateur: {msg['question']}")
                formatted_messages.append(f"Assistant: {msg['response']}")

        return "\n".join(formatted_messages) if formatted_messages else "Aucun historique disponible."

    def generate_response(self, discussion_id: str, question: str, context_id: str):
        """Génère une réponse en utilisant le résumé et l'historique"""

        # 🔹 Vérifier si l'ID de discussion est valide
        if not isinstance(discussion_id, str) or len(discussion_id) != 24:
            return {"error": "L'ID de la discussion est invalide"}

        # 🔹 Charger le contexte
        context = context_service.get_context(context_id)
        if not context:
            return {"error": "Contexte non trouvé"}

        # 🔹 Récupérer le résumé de la discussion
        summary = discussion_service.get_summary(discussion_id) or "Aucun résumé disponible."

        # 🔹 Récupérer l'historique de la discussion
        history = history_service.get_history_by_discussion(discussion_id)
        formatted_messages = self.format_messages(history)

        # 🔹 Construire le prompt simple
        prompt = (
            f"💡 **Résumé de la discussion** :\n{summary}\n\n"
            # f"📜 **Historique des échanges** :\n{formatted_messages}\n\n"
            f"📝 **Nouvelle question** :\nUtilisateur : {question}\nAssistant :"
        )

        print('-----------------------------------------------------------')
        print(f'Prompt envoyé à DeepSeek :\n{prompt}')
        print('-----------------------------------------------------------')

        # 🔹 Envoyer la requête à Ollama
        response = requests.post(
            self.url,
            json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": False}
        )

        if response.status_code != 200:
            return {"error": f"Erreur avec Ollama: {response.text}"}

        data = response.json()
        answer = data.get("response", "")

        # ✅ Ajouter l'ID de discussion dans l'historique
        history_service.add_history(discussion_id, question, answer, context_id)

        # ✅ Mettre à jour le résumé avec l'ID de la discussion
        discussion_service.update_summary(discussion_id, question, answer)

        return {"response": answer}


# Singleton
generate_service = GenerateService()
