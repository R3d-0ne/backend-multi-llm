import logging
import json
import requests
import os
import traceback
from typing import Dict, Any, Optional, List
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    """
    Service pour interagir avec les modèles de langage (LLM).
    Permet d'envoyer des prompts et de récupérer des réponses générées.
    Supporte la sélection dynamique de modèles.
    """

    def __init__(self):
        # Configuration du modèle par défaut
        self.base_url = os.getenv("LLM_API_URL", "http://host.docker.internal:8000")
        self.model_name = os.getenv("LLM_MODEL_NAME", "llama3.1:8b")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.request_timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "10"))  # Timeout en secondes

        # Liste des modèles disponibles
        self.available_models = self._get_available_models()

        logger.info(
            f"Service LLM initialisé avec le modèle '{self.model_name}' à l'URL '{self.base_url}' (timeout: {self.request_timeout}s)")

    def _get_available_models(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des modèles disponibles depuis le serveur LLM

        Returns:
            Liste des modèles avec leurs caractéristiques
        """
        models = []
        try:
            # Essayer de récupérer les modèles depuis l'API
            if "host.docker.internal" in self.base_url or "127.0.0.1" in self.base_url:
                # Format pour Ollama
                api_url = f"{self.base_url}/api/tags"
                response = requests.get(api_url, timeout=self.request_timeout)
                if response.status_code == 200:
                    data = response.json()
                    if "models" in data:
                        for model in data["models"]:
                            models.append({
                                "id": model.get("name", ""),
                                "name": model.get("name", "").split(":")[0],
                                "size": model.get("size", 0),
                                "provider": "ollama"
                            })

            # Si aucun modèle n'est disponible, on utilise une liste par défaut
            if not models:
                default_models = [
                    {"id": "llama3.1:8b", "name": "Llama 3.1 (8B)", "provider": "ollama"},
                    {"id": "llama3.1:70b", "name": "Llama 3.1 (70B)", "provider": "ollama"},
                    {"id": "mistral:7b", "name": "Mistral (7B)", "provider": "ollama"},
                    {"id": "phi3:mini", "name": "Phi-3 Mini", "provider": "ollama"}
                ]
                models = default_models

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles disponibles: {e}")
            # Liste par défaut en cas d'erreur
            models = [
                {"id": "llama3.1:8b", "name": "Llama 3.1 (8B)", "provider": "ollama"},
                {"id": "llama3.1:70b", "name": "Llama 3.1 (70B)", "provider": "ollama"},
                {"id": "mistral:7b", "name": "Mistral (7B)", "provider": "ollama"},
                {"id": "phi3:mini", "name": "Phi-3 Mini", "provider": "ollama"}
            ]

        return models

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Renvoie la liste des modèles disponibles pour l'interface utilisateur

        Returns:
            Liste des modèles avec leurs caractéristiques
        """
        return self.available_models

    def set_model(self, model_id: str) -> bool:
        """
        Change le modèle utilisé par le service

        Args:
            model_id: L'identifiant du modèle à utiliser

        Returns:
            True si le changement a réussi, False sinon
        """
        # Vérifier que le modèle est dans la liste des disponibles
        model_exists = any(model["id"] == model_id for model in self.available_models)

        if model_exists or model_id in [m["id"] for m in self.available_models]:
            self.model_name = model_id
            logger.info(f"Modèle changé pour: {model_id}")
            return True
        else:
            logger.warning(f"Modèle {model_id} non disponible")
            return False

    def generate_response(self, prompt: str,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          system_prompt: Optional[str] = None,
                          model_override: Optional[str] = None) -> str:
        """
        Génère une réponse à partir d'un prompt en utilisant le LLM.

        Args:
            prompt: Le texte du prompt à envoyer au modèle
            temperature: Température pour le sampling (plus basse = plus déterministe)
            max_tokens: Nombre maximum de tokens à générer
            system_prompt: Message système pour guider le modèle (chat models only)
            model_override: Utiliser un modèle spécifique pour cette requête

        Returns:
            La réponse générée par le modèle
        """
        # Utiliser le modèle spécifié pour cette requête si fourni
        current_model = model_override if model_override else self.model_name

        for attempt in range(1, 3):  # Max 2 tentatives
            try:
                # Utiliser les valeurs par défaut si non spécifiées
                temperature = temperature if temperature is not None else self.temperature
                max_tokens = max_tokens if max_tokens is not None else self.max_tokens

                # Construire les paramètres de l'API en fonction du type de modèle
                if "llama" in current_model.lower() or "mistral" in current_model.lower() or "phi" in current_model.lower():
                    # Pour Ollama, utiliser l'API de chat
                    if "host.docker.internal" in self.base_url or "127.0.0.1" in self.base_url:
                        # Format pour l'API Ollama
                        system_message = system_prompt or "Vous êtes un assistant d'extraction d'information précis qui répond uniquement au format demandé."

                        payload = {
                            "model": current_model,
                            "messages": [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        }
                        api_url = f"{self.base_url}/api/chat"
                    else:
                        # Format pour les modèles chat (API similaire à OpenAI)
                        payload = {
                            "model": current_model,
                            "messages": [
                                {"role": "system",
                                 "content": system_prompt or "Vous êtes un assistant "},
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                        api_url = f"{self.base_url}/api/generate"

                    logger.info(f"Tentative {attempt}/2: Envoi de requête au modèle chat {current_model}")
                    response = requests.post(api_url, json=payload, timeout=self.request_timeout)
                    response.raise_for_status()

                    # Extraire le contenu du message de réponse
                    response_data = response.json()

                    # Format API chat d'Ollama
                    if "message" in response_data and "content" in response_data["message"]:
                        content = response_data["message"]["content"]
                        logger.info(f"Réponse Ollama Chat reçue ({len(content)} caractères)")
                        return content
                    # Format API generate d'Ollama
                    elif "response" in response_data:
                        content = response_data.get("response", "")
                        if not content and response_data.get("done_reason") == "load":
                            logger.warning("Le modèle a été chargé mais n'a pas généré de réponse")
                            if attempt < 2:
                                logger.info("Nouvelle tentative après chargement du modèle...")
                                time.sleep(1)
                                continue
                            else:
                                return self._get_fallback_response(
                                    "Le modèle a été chargé mais n'a pas généré de réponse")

                        logger.info(f"Réponse Ollama Generate reçue ({len(content)} caractères)")
                        return content
                    # Format OpenAI
                    elif "choices" in response_data and len(response_data["choices"]) > 0:
                        content = response_data["choices"][0]["message"]["content"]
                        logger.info(f"Réponse OpenAI reçue ({len(content)} caractères)")
                        return content
                    else:
                        logger.error(f"Format de réponse inattendu: {response_data}")
                        if attempt < 2:
                            logger.info("Format inconnu, nouvelle tentative...")
                            time.sleep(1)
                            continue
                        return self._get_fallback_response("Format de réponse inattendu")

                else:
                    # Format pour les modèles de complétion (ancienne API)
                    if "host.docker.internal" in self.base_url or "127.0.0.1" in self.base_url:
                        # Format pour l'API Ollama
                        payload = {
                            "model": current_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        }
                        api_url = f"{self.base_url}/api/generate"
                    else:
                        # Format pour l'API de complétion standard
                        payload = {
                            "model": current_model,
                            "prompt": prompt,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                        api_url = f"{self.base_url}/api/generate"

                    logger.info(f"Tentative {attempt}/2: Envoi de requête au modèle de complétion {current_model}")
                    response = requests.post(api_url, json=payload, timeout=self.request_timeout)
                    response.raise_for_status()

                    # Extraire la réponse
                    response_data = response.json()

                    # Format API chat d'Ollama
                    if "message" in response_data and "content" in response_data["message"]:
                        content = response_data["message"]["content"]
                        logger.info(f"Réponse Ollama Chat reçue ({len(content)} caractères)")
                        return content
                    # Format API generate d'Ollama
                    elif "response" in response_data:
                        content = response_data.get("response", "")
                        if not content and response_data.get("done_reason") == "load":
                            logger.warning("Le modèle a été chargé mais n'a pas généré de réponse")
                            if attempt < 2:
                                logger.info("Nouvelle tentative après chargement du modèle...")
                                time.sleep(1)
                                continue
                            else:
                                return self._get_fallback_response(
                                    "Le modèle a été chargé mais n'a pas généré de réponse")

                        logger.info(f"Réponse Ollama Generate reçue ({len(content)} caractères)")
                        return content
                    # Format API de complétion classique
                    elif "text" in response_data:
                        logger.info(f"Réponse texte reçue ({len(response_data['text'])} caractères)")
                        return response_data["text"]
                    else:
                        logger.error(f"Format de réponse inattendu: {response_data}")
                        if attempt < 2:
                            logger.info("Format inconnu, nouvelle tentative...")
                            time.sleep(1)
                            continue
                        return self._get_fallback_response("Format de réponse inattendu")

            except requests.exceptions.Timeout:
                logger.error(f"Timeout lors de la connexion au service LLM après {self.request_timeout} secondes")
                if attempt < 2:
                    logger.info(f"Nouvelle tentative dans 2 secondes...")
                    time.sleep(2)
                else:
                    return self._get_fallback_response("Le service LLM n'est pas disponible (timeout)")

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Erreur de connexion au service LLM: {e}")
                if attempt < 2:
                    logger.info(f"Nouvelle tentative dans 2 secondes...")
                    time.sleep(2)
                else:
                    return self._get_fallback_response("Le service LLM n'est pas accessible")

            except Exception as e:
                logger.error(f"Erreur lors de la génération de réponse: {e}")
                logger.error(traceback.format_exc())
                return self._get_fallback_response(f"Erreur: {str(e)}")

        # Si on arrive ici, c'est que toutes les tentatives ont échoué
        return self._get_fallback_response("Toutes les tentatives de connexion au LLM ont échoué")

    def _get_fallback_response(self, error_message: str) -> str:
        """Fournit une réponse de secours en cas d'échec du LLM"""
        logger.warning(f"Utilisation de la réponse de secours: {error_message}")
        return json.dumps({
            "error": True,
            "message": error_message,
            "fallback": True,
            "extracted_data": {}
        })

    def extract_structured_data(self, text: str, schema: Dict[str, Any], model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Extrait des données structurées à partir d'un texte selon un schéma donné.

        Args:
            text: Le texte à analyser
            schema: Un dictionnaire définissant le schéma des données à extraire
            model_override: Modèle spécifique à utiliser pour cette extraction

        Returns:
            Un dictionnaire contenant les données extraites selon le schéma
        """
        try:
            # Construire un prompt pour extraire les données selon le schéma
            schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

            prompt = f"""
            TÂCHE: Extraire des informations structurées du texte suivant selon le schéma fourni.
            FORMAT DE RÉPONSE: JSON uniquement, sans texte avant ou après.

            TEXTE:
            {text[:3000]}  # Limiter la taille pour éviter les dépassements de contexte

            SCHÉMA:
            {schema_str}

            Renvoyer un objet JSON conforme au schéma, contenant uniquement les informations extraites du texte.
            """

            # Générer la réponse avec le modèle spécifié si fourni
            response = self.generate_response(
                prompt=prompt,
                temperature=0.1,  # Température basse pour des résultats cohérents
                system_prompt="Vous êtes un assistant d'extraction de données précis. Extrayez uniquement les informations demandées, au format JSON exact.",
                model_override=model_override
            )

            # Vérifier si nous avons une réponse d'erreur
            try:
                response_obj = json.loads(response)
                if isinstance(response_obj, dict) and response_obj.get("error", False):
                    logger.warning("Réponse d'erreur reçue du LLM")
                    return response_obj
            except:
                pass

            # Extraire le JSON de la réponse
            try:
                # Essayer d'abord de parser directement la réponse
                return json.loads(response)
            except json.JSONDecodeError:
                # Si échec, chercher un bloc JSON dans la réponse
                import re
                json_blocks = re.findall(r'```(?:json)?([\s\S]*?)```', response)
                if json_blocks:
                    return json.loads(json_blocks[0].strip())

                # Chercher quelque chose entre accolades
                json_matches = re.findall(r'({[\s\S]*?})', response)
                if json_matches:
                    return json.loads(json_matches[0])

                # Si aucune méthode ne fonctionne
                logger.error(f"Impossible d'extraire un JSON valide de la réponse (longueur: {len(response)} caractères)")
                return {}

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de données structurées: {e}")
            logger.error(traceback.format_exc())
            return {}


# Singleton
llm_service = LLMService()