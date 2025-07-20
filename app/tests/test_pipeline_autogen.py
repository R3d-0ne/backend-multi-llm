import os
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration du LLM
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

# Configuration Ollama
config_list = [
    {
        "model": LLM_MODEL,
        "base_url": "http://host.docker.internal:11434",
        "api_type": "ollama"
    }
]

# Création de l'assistant
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "cache_seed": 41,  # seed pour le cache et la reproductibilité
        "config_list": config_list,
        "temperature": 0,  # température pour le sampling
    },
    system_message="""Tu es un assistant expert en Python.
    Quand tu génères du code Python :
    1. Inclus TOUS les imports nécessaires au début
    2. Utilise des noms de variables explicites
    3. Ajoute des commentaires pour expliquer le code
    4. Gère les erreurs potentielles avec try/except
    5. N'essaie PAS d'installer des packages dans le code
    7. N'utilise que des caractères ASCII dans les noms de variables et commentaires
    8. N'utilise pas de caractères spéciaux dans les noms de variables et commentaires
    9. Termine par '#TERMINATE' quand la tâche est terminée

    """
)

# Création du UserProxyAgent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        # the executor to run the generated code
        "executor": LocalCommandLineCodeExecutor(work_dir="outputs"),
    },
)

# Tâche à exécuter
def run_task(task_message):
    # L'assistant reçoit un message du user_proxy contenant la description de la tâche
    chat_res = user_proxy.initiate_chat(
        assistant,
        message=task_message,
        summary_method="reflection_with_llm",
    )
    return chat_res

if __name__ == "__main__":
    # Exemple de tâche
    task = """Créer un script Python qui :
  qui crée un jeu de la vie.
    Sauvegarde le code dans un fichier random_stats.py dans le dossier outputs.
    """
    run_task(task)