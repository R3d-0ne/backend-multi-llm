import uuid
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class Discussion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Identifiant unique de la discussion")
    title: str = Field(..., min_length=1, description="Titre de la discussion")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Date de création de la discussion")


    model_config = {
        "populate_by_name": True,
        "strip_whitespace": True,
    }
