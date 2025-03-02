import uuid
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class Settings(BaseModel):
    # Utilisation d'un ID fixe pour s'assurer qu'il n'y ait qu'un seul document de paramètres.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Identifiant fixe pour les paramètres")
    title: str = Field(..., min_length=1, description="Titre des paramètres")
    content: str = Field(..., min_length=1, description="Contenu détaillé des paramètres")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Date de mise à jour des paramètres")



    model_config = {
        "populate_by_name": True,
        "strip_whitespace": True,
    }
