import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class Context(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identifiant unique du prompt contextuel"
    )
    discussion_id: str = Field(
        ...,
        description="Identifiant de la discussion associée"
    )
    setting_id: Optional[str] = Field(
        None,
        description="Identifiant des paramètres de la discussion (optionnel)"
    )
    current_message: str = Field(
        ...,
        min_length=1,
        description="Message actuel ou requête en cours"
    )
    history: Optional[List[str]] = Field(
        default_factory=list,
        description="Historique des échanges précédents (chaque élément représente un échange, ex: 'Utilisateur: ... / Assistant: ...')"
    )
    additional_info: Optional[str] = Field(
        None,
        description="Informations complémentaires éventuelles (par exemple, contexte spécifique, instructions particulières)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date et heure de création du prompt"
    )

    @field_validator("id", mode="before")
    def validate_uuid(cls, v):
        try:
            uuid.UUID(str(v))
            return str(v)
        except Exception as e:
            raise ValueError("L'id doit être un UUID valide") from e

    @field_validator("discussion_id", mode="before")
    def validate_discussion_id(cls, v):
        try:
            uuid.UUID(str(v))
            return str(v)
        except Exception as e:
            raise ValueError("Le discussion_id doit être un UUID valide") from e

    @field_validator("setting_id", mode="before")
    def validate_setting_id(cls, v):
        if v is None:
            return v
        try:
            uuid.UUID(str(v))
            return str(v)
        except Exception as e:
            raise ValueError("Le setting_id doit être un UUID valide") from e

    model_config = {
        "populate_by_name": True,
        "strip_whitespace": True,
    }
