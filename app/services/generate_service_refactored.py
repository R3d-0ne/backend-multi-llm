"""
Service de génération refactorisé avec validation améliorée et gestion d'erreurs robuste.
"""
import logging
from typing import Optional, Dict, Any
from pydantic import BaseModel

from .base_service import BaseService, LogLevel, ServiceResponse, ValidationError, ExternalServiceError
from .config_service import config_service
from .context_service import context_service
from .discussions_service import discussions_service
from .message_service import message_service
from .settings_service import settings_service
from .llm_service import llm_service
from ..models.message import Message


class GenerateRequest(BaseModel):
    """Modèle de requête pour la génération"""
    discussion_id: Optional[str] = None
    settings_id: Optional[str] = None
    current_message: str
    additional_info: Optional[str] = None
    model_id: Optional[str] = None


class GenerateServiceRefactored(BaseService):
    """
    Service de génération refactorisé avec orchestration améliorée et validation robuste.
    """

    def __init__(self):
        super().__init__("GenerateService")
        self.config = config_service.llm
        
        self.log(LogLevel.INFO, "Service de génération initialisé")

    def generate_response(
        self,
        discussion_id: Optional[str],
        settings_id: Optional[str],
        current_message: str,
        additional_info: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> ServiceResponse:
        """
        Orchestre la génération d'une réponse avec validation et gestion d'erreurs améliorées.

        Args:
            discussion_id: ID de la discussion (créé automatiquement si None)
            settings_id: ID des paramètres à utiliser
            current_message: Message de l'utilisateur
            additional_info: Informations supplémentaires optionnelles
            model_id: ID du modèle LLM à utiliser

        Returns:
            ServiceResponse contenant la réponse générée
        """
        # Validation des entrées
        self.validate_required_fields(
            {"current_message": current_message},
            ["current_message"]
        )
        
        self.validate_field_type(current_message, str, "current_message")
        
        if not current_message.strip():
            return ServiceResponse.error_response(
                "Le message ne peut pas être vide",
                "EMPTY_MESSAGE"
            )

        def generation_operation():
            return self._execute_generation(
                discussion_id, settings_id, current_message,
                additional_info, model_id
            )

        return self.safe_execute(
            generation_operation,
            "Erreur lors de la génération de la réponse"
        )

    def _execute_generation(
        self,
        discussion_id: Optional[str],
        settings_id: Optional[str],
        current_message: str,
        additional_info: Optional[str],
        model_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Exécute la logique de génération.
        """
        self.log(LogLevel.INFO, f"Génération de réponse pour le message: {current_message[:50]}...")

        # Étape 1: Gestion de la discussion
        discussion_id = self._ensure_discussion_exists(discussion_id)

        # Étape 2: Récupération des paramètres
        generation_settings = self._get_generation_settings(settings_id)

        # Étape 3: Construction du contexte
        context_result = self._build_context(
            discussion_id, current_message, additional_info, settings_id
        )

        # Étape 4: Génération de la réponse
        answer = self._generate_with_llm(
            context_result["prompt"],
            generation_settings,
            model_id
        )

        # Étape 5: Enregistrement des messages
        self._save_messages(discussion_id, current_message, answer)

        return {
            "response": answer,
            "context_id": context_result["id"],
            "discussion_id": discussion_id,
            "settings_used": generation_settings,
            "model_used": model_id or self.config.model
        }

    def _ensure_discussion_exists(self, discussion_id: Optional[str]) -> str:
        """
        S'assure qu'une discussion existe, en crée une si nécessaire.
        
        Args:
            discussion_id: ID de discussion optionnel
            
        Returns:
            ID de la discussion (existante ou nouvellement créée)
        """
        if discussion_id:
            self.log(LogLevel.DEBUG, f"Utilisation de la discussion existante: {discussion_id}")
            return discussion_id

        # Créer une nouvelle discussion
        self.log(LogLevel.INFO, "Création d'une nouvelle discussion")
        new_discussion = discussions_service.add_discussion("Nouvelle discussion")
        
        if not new_discussion or "id" not in new_discussion:
            raise ExternalServiceError(
                "Impossible de créer une nouvelle discussion",
                "DiscussionService"
            )

        new_discussion_id = new_discussion["id"]
        self.log(LogLevel.INFO, f"Nouvelle discussion créée: {new_discussion_id}")
        
        return new_discussion_id

    def _get_generation_settings(self, settings_id: Optional[str]) -> Dict[str, Any]:
        """
        Récupère les paramètres de génération.
        
        Args:
            settings_id: ID des paramètres optionnel
            
        Returns:
            Dictionnaire des paramètres de génération
        """
        default_settings = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "system_prompt": None
        }

        if not settings_id:
            self.log(LogLevel.DEBUG, "Utilisation des paramètres par défaut")
            return default_settings

        try:
            self.log(LogLevel.DEBUG, f"Récupération des paramètres: {settings_id}")
            settings_data = settings_service.get_settings_by_id(settings_id)
            
            if not settings_data:
                self.log(LogLevel.WARNING, f"Paramètres non trouvés: {settings_id}, utilisation des valeurs par défaut")
                return default_settings

            # Extraire les paramètres depuis le payload
            if "payload" in settings_data and isinstance(settings_data["payload"], dict):
                payload = settings_data["payload"]
                return {
                    "temperature": payload.get("temperature", default_settings["temperature"]),
                    "max_tokens": payload.get("max_tokens", default_settings["max_tokens"]),
                    "system_prompt": payload.get("system_prompt", default_settings["system_prompt"])
                }
            else:
                self.log(LogLevel.WARNING, "Format de paramètres invalide, utilisation des valeurs par défaut")
                return default_settings

        except Exception as e:
            self.log(LogLevel.WARNING, f"Erreur lors de la récupération des paramètres: {e}")
            return default_settings

    def _build_context(
        self,
        discussion_id: str,
        current_message: str,
        additional_info: Optional[str],
        settings_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Construit le contexte pour la génération.
        
        Args:
            discussion_id: ID de la discussion
            current_message: Message actuel
            additional_info: Informations supplémentaires
            settings_id: ID des paramètres
            
        Returns:
            Dictionnaire contenant le contexte et l'ID de contexte
        """
        try:
            self.log(LogLevel.DEBUG, f"Construction du contexte pour la discussion: {discussion_id}")
            
            # Appel du service de contexte pour construire et enregistrer le prompt final
            context_result = context_service.save_full_context(
                discussion_id=discussion_id,
                current_message=current_message,
                additional_info=additional_info,
                settings_id=settings_id
            )
            
            if not context_result or "id" not in context_result:
                raise ExternalServiceError(
                    "Impossible de sauvegarder le contexte",
                    "ContextService"
                )

            context_id = context_result["id"]

            # Récupérer le prompt stocké
            prompt_data = context_service.get_context(context_id)
            
            if not prompt_data or "prompt" not in prompt_data:
                raise ExternalServiceError(
                    "Impossible de récupérer le prompt du contexte",
                    "ContextService"
                )

            prompt = prompt_data["prompt"]
            self.log(LogLevel.DEBUG, f"Prompt construit (longueur: {len(prompt)} caractères)")

            return {
                "id": context_id,
                "prompt": prompt,
                "discussion_id": discussion_id
            }

        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors de la construction du contexte: {e}")
            raise

    def _generate_with_llm(
        self,
        prompt: str,
        settings: Dict[str, Any],
        model_override: Optional[str]
    ) -> str:
        """
        Génère une réponse en utilisant le service LLM.
        
        Args:
            prompt: Prompt à envoyer au LLM
            settings: Paramètres de génération
            model_override: Modèle spécifique à utiliser
            
        Returns:
            Réponse générée par le LLM
        """
        try:
            self.log(LogLevel.DEBUG, f"Génération LLM avec le modèle: {model_override or self.config.model}")
            
            # Utiliser le service LLM pour générer une réponse
            answer = llm_service.generate_response(
                prompt=prompt,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                system_prompt=settings["system_prompt"],
                model_override=model_override
            )

            if not answer:
                raise ExternalServiceError(
                    "Réponse vide du service LLM",
                    "LLMService"
                )

            self.log(LogLevel.INFO, f"Réponse générée avec succès (longueur: {len(answer)} caractères)")
            return answer

        except Exception as e:
            self.log(LogLevel.ERROR, f"Erreur lors de la génération LLM: {e}")
            raise ExternalServiceError(
                f"Erreur du service LLM: {str(e)}",
                "LLMService"
            )

    def _save_messages(self, discussion_id: str, user_message: str, assistant_response: str) -> None:
        """
        Enregistre les messages de l'utilisateur et de l'assistant.
        
        Args:
            discussion_id: ID de la discussion
            user_message: Message de l'utilisateur
            assistant_response: Réponse de l'assistant
        """
        try:
            # Enregistrer le message de l'utilisateur
            self.log(LogLevel.DEBUG, "Enregistrement du message utilisateur")
            user_msg = Message(
                discussion_id=discussion_id,
                sender="user",
                text=user_message
            )
            message_service.send_message(user_msg)

            # Enregistrer la réponse de l'assistant
            self.log(LogLevel.DEBUG, "Enregistrement de la réponse assistant")
            assistant_msg = Message(
                discussion_id=discussion_id,
                sender="assistant",
                text=assistant_response
            )
            message_service.send_message(assistant_msg)

            self.log(LogLevel.INFO, f"Messages enregistrés pour la discussion: {discussion_id}")

        except Exception as e:
            # Log l'erreur mais ne pas faire échouer la génération
            self.log(LogLevel.WARNING, f"Erreur lors de l'enregistrement des messages: {e}")

    def validate_generate_request(self, request_data: Dict[str, Any]) -> ServiceResponse:
        """
        Valide une requête de génération complète.
        
        Args:
            request_data: Données de la requête
            
        Returns:
            ServiceResponse indiquant si la validation a réussi
        """
        try:
            # Valider avec le modèle Pydantic
            validated_request = GenerateRequest(**request_data)
            
            # Validations supplémentaires
            if not validated_request.current_message.strip():
                return ServiceResponse.error_response(
                    "Le message ne peut pas être vide",
                    "EMPTY_MESSAGE"
                )

            if validated_request.discussion_id and not isinstance(validated_request.discussion_id, str):
                return ServiceResponse.error_response(
                    "ID de discussion invalide",
                    "INVALID_DISCUSSION_ID"
                )

            if validated_request.settings_id and not isinstance(validated_request.settings_id, str):
                return ServiceResponse.error_response(
                    "ID de paramètres invalide",
                    "INVALID_SETTINGS_ID"
                )

            return ServiceResponse.success_response({
                "valid": True,
                "validated_data": validated_request.dict()
            })

        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur de validation: {str(e)}",
                "VALIDATION_ERROR",
                {"original_data": request_data}
            )

    def get_generation_statistics(self) -> ServiceResponse:
        """
        Obtient des statistiques sur les générations.
        
        Returns:
            ServiceResponse contenant les statistiques
        """
        def stats_operation():
            # Pour l'instant, retourner des statistiques basiques
            # Dans une version future, on pourrait tracker plus de métriques
            return {
                "service": "GenerateService",
                "model_default": self.config.model,
                "temperature_default": self.config.temperature,
                "max_tokens_default": self.config.max_tokens,
                "llm_base_url": self.config.base_url
            }

        return self.safe_execute(
            stats_operation,
            "Erreur lors de la récupération des statistiques"
        )

    def health_check(self) -> ServiceResponse:
        """
        Vérifie l'état de santé du service de génération.
        
        Returns:
            ServiceResponse avec l'état de santé
        """
        health_data = {
            "status": "healthy",
            "service": "GenerateService",
            "dependencies": {}
        }

        try:
            # Vérifier les dépendances
            dependencies = [
                ("ContextService", context_service),
                ("DiscussionsService", discussions_service),
                ("MessageService", message_service),
                ("SettingsService", settings_service),
                ("LLMService", llm_service)
            ]

            all_healthy = True
            
            for dep_name, dep_service in dependencies:
                try:
                    # Test basique de chaque dépendance
                    if hasattr(dep_service, 'health_check'):
                        dep_health = dep_service.health_check()
                        health_data["dependencies"][dep_name] = {
                            "status": "healthy" if dep_health else "unhealthy"
                        }
                    else:
                        # Si pas de health_check, considérer comme disponible
                        health_data["dependencies"][dep_name] = {
                            "status": "available"
                        }
                except Exception as e:
                    health_data["dependencies"][dep_name] = {
                        "status": "error",
                        "error": str(e)
                    }
                    all_healthy = False

            health_data["overall_status"] = "healthy" if all_healthy else "degraded"
            
            if all_healthy:
                return ServiceResponse.success_response(health_data)
            else:
                return ServiceResponse.error_response(
                    "Certaines dépendances ne sont pas disponibles",
                    "DEPENDENCIES_UNAVAILABLE",
                    health_data
                )

        except Exception as e:
            return ServiceResponse.error_response(
                f"Erreur lors du test de santé: {str(e)}",
                "HEALTH_CHECK_ERROR"
            )


# Instance du service de génération refactorisé
generate_service_refactored = GenerateServiceRefactored()