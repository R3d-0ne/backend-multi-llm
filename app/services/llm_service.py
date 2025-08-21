"""
Service LLM pour gérer les interactions avec les modèles de langage.
Actuellement configuré pour Ollama, peut être étendu pour d'autres providers.
"""
import logging
import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Configuration du logging
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()


class LLMService:
    """Service pour gérer les interactions avec les modèles LLM."""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.current_model = os.getenv("DEFAULT_MODEL", "qwen2.5:7b")
        self.timeout = 60
        logger.info(f"Service LLM initialisé - URL: {self.base_url}, Modèle: {self.current_model}")
    
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
    
    def generate_response(
        self, 
        prompt: str, 
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Génère une réponse à partir d'un prompt."""
        try:
            model_to_use = model_id if model_id else self.current_model
            
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                payload["options"]["num_predict"] = max_tokens
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logger.info(f"Réponse générée avec succès (modèle: {model_to_use})")
            return generated_text
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            return f"Erreur lors de la génération de la réponse: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé du service LLM."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "url": self.base_url,
                "current_model": self.current_model,
                "response_time": response.elapsed.total_seconds()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "url": self.base_url,
                "current_model": self.current_model,
                "error": str(e)
            }


# Instance globale du service LLM
llm_service = LLMService()