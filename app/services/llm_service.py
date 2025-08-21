"""
Service LLM basique pour la génération de réponses.
Fournit une interface unifiée pour les appels aux modèles de langue.
"""
import logging
import requests
from typing import Dict, Any, Optional

from .config_service import config_service

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service basique pour les appels aux modèles de langue (LLM).
    """
    
    def __init__(self):
        logger.info(f"Service LLM initialisé avec l'URL: {config_service.llm.base_url}")

    
    def generate_response(
        self, 
        prompt: str, 
        model: Optional[str] = None, 
        temperature: float = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Génère une réponse avec le modèle LLM.
        
        Args:
            prompt: Le prompt à envoyer au modèle
            model: Le modèle à utiliser (optionnel)
            temperature: La température pour la génération (optionnel)
            max_tokens: Le nombre maximum de tokens
            
        Returns:
            La réponse générée par le modèle
            
        Raises:
            Exception: En cas d'erreur lors de l'appel au modèle
        """
        if not model:
            model = config_service.llm.model
        
        if temperature is None:
            temperature = config_service.llm.temperature
            
        try:
            # Construction de la requête pour Ollama
            url = f"{config_service.llm.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            response = requests.post(url, json=payload, timeout=config_service.llm.timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de l'appel au LLM: {e}")
            raise Exception(f"Erreur LLM: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue dans le service LLM: {e}")
            raise Exception(f"Erreur LLM: {e}")
    
    def is_available(self) -> bool:
        """
        Vérifie si le service LLM est disponible.
        
        Returns:
            True si le service est disponible, False sinon
        """
        try:
            response = requests.get(f"{config_service.llm.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
					
					
					
		 def get_available_models(self) -> List[Dict[str, Any]]:
        """Récupère la liste des modèles disponibles."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            models_data = response.json()
            models = []

            for model in models_data.get("models", []):
                models.append({
                    "id": model.get("name", ""),
                    "name": model.get("name", ""),
                    "size": model.get("size", 0),
                    "digest": model.get("digest", ""),
                    "modified_at": model.get("modified_at", "")
                })

            logger.info(f"Trouvé {len(models)} modèles disponibles")
            return models

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {e}")
            # Retourner un modèle par défaut en cas d'erreur
            return [{
                "id": self.current_model,
                "name": self.current_model,
                "size": 0,
                "digest": "",
                "modified_at": ""
            }]

    def set_model(self, model_id: str) -> bool:
        """Définit le modèle à utiliser."""
        try:
            # Vérifier que le modèle existe
            available_models = self.get_available_models()
            model_ids = [m["id"] for m in available_models]

            if model_id in model_ids:
                self.current_model = model_id
                logger.info(f"Modèle défini: {model_id}")
                return True
            else:
                logger.warning(f"Modèle {model_id} non trouvé. Modèles disponibles: {model_ids}")
                return False

        except Exception as e:
            logger.error(f"Erreur lors de la définition du modèle: {e}")
            return False
					
					
					
					
			
					
		


# Instance singleton du service LLM

llm_service = LLMService()