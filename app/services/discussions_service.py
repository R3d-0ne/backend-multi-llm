import re
import requests
from .mongo_service import mongo_service
from datetime import datetime
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")


class DiscussionService:
    def __init__(self):
        self.collection_name = "discussions"

    def create_discussion(self, context_id: str) -> str:
        """ Crée une nouvelle discussion et retourne son ID """
        discussion = {
            "context_id": context_id,
            "summary": "",  # 🔹 Commence avec un résumé vide
            "created_at": datetime.utcnow()
        }
        return mongo_service.insert_document(self.collection_name, discussion)

    def get_summary(self, discussion_id: str) -> str:
        """ Récupère le résumé actuel d'une discussion """
        discussion = mongo_service.get_document_by_id(self.collection_name, discussion_id)
        return discussion["summary"] if discussion else ""

    def update_summary(self, discussion_id, question, answer):
        """ Met à jour le résumé en générant un nouveau texte condensé """

        # 🔹 Vérifier si l'ID de la discussion est valide
        discussion = mongo_service.get_document_by_id(self.collection_name, discussion_id)
        if not discussion:
            raise RuntimeError(f"❌ Discussion introuvable : {discussion_id}")

        previous_summary = discussion.get("summary", "") if discussion else ""

        # 🔹 Construire le texte à résumer
        text_to_summarize = f"{previous_summary}\nUtilisateur : {question}\nAssistant : {answer}"

        # 🔹 Générer un nouveau résumé
        new_summary = self.summarize_text(previous_summary, answer)

        # 🔹 Vérifier si le résumé est bien généré
        if not new_summary or len(new_summary.strip()) == 0:
            raise RuntimeError(f"❌ Le résumé généré est vide pour la discussion {discussion_id}")

        print(f"✅ Nouveau résumé généré :\n{new_summary}")

        # 🔹 Mettre à jour la discussion avec le nouveau résumé
        try:
            update_result = mongo_service.update_document(self.collection_name, discussion_id, {"summary": new_summary})
            if not update_result:
                raise RuntimeError(
                    f"❌ Échec de la mise à jour du résumé dans MongoDB pour la discussion {discussion_id}"
                )
        except Exception as e:
            raise RuntimeError(f"❌ Erreur MongoDB lors de l'update de la discussion {discussion_id}: {str(e)}")

        return new_summary

    def summarize_text(self, previous_summary, new_text):
        """ Génère un résumé mis à jour en intégrant les nouvelles discussions """

        prompt = (
            "🎯 Objectif : Générer un résumé **court et synthétique** de la discussion en cours, "
            "en tenant compte de l'historique précédent que tu mettras à jour au fur et à mesure.\n"
            "🔹 Intègre les nouvelles informations de façon **fluide** sans répétition inutile.\n"
            "🔹 Utilise des phrases courtes, directes et simples.\n"
            "🔹 Ne réfléchis pas à la tâche et ne pose pas de questions, génère uniquement le résumé.\n\n"
            f"📝 **Résumé précédent** :\n{previous_summary}\n\n"
            f"💬 **Nouvelle discussion** :\n{new_text}\n\n"
            "⚡ **Résumé mis à jour** :"
        )

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "deepseek-r1:7b", "prompt": prompt, "stream": False}
        )

        if response.status_code != 200:
            print(f"⚠️ Erreur API Ollama: {response.text}")
            return previous_summary  # 🔹 Si échec, on garde l'ancien résumé

        data = response.json()
        raw_summary = data.get("response", "").strip()

        # 🔥 Nettoyage : Suppression des balises <think> et du contenu inutile
        cleaned_summary = re.sub(r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL).strip()

        return cleaned_summary

    def delete_discussion(self, discussion_id: str) -> bool:
        """ Supprime une discussion par son ID """
        discussion = mongo_service.get_document_by_id(self.collection_name, discussion_id)

        if not discussion:
            print(f"❌ Aucune discussion trouvée avec l'ID {discussion_id}")
            return False

        delete_result = mongo_service.delete_document(self.collection_name, discussion_id)

        if delete_result:
            print(f"✅ Discussion supprimée avec succès : {discussion_id}")
            return True
        else:
            print(f"❌ Échec de la suppression de la discussion {discussion_id}")
            return False

    def get_all_discussions(self):
        discussions = mongo_service.get_all_documents(self.collection_name)

        if discussions:
            print(f"✅ {len(discussions)} discussions trouvées")
        else:
            print("❌ Aucune discussion trouvée")

        return discussions or []  # Retourne un tableau vide si discussions est `None` ou vide


# Singleton
discussion_service = DiscussionService()
