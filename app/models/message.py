import uuid
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Identifiant unique du message")
    discussion_id: str = Field(..., description="ID de la discussion associée")
    sender: str = Field(..., description="Expéditeur du message, 'user' ou 'assistant'")
    text: str = Field(..., min_length=1, description="Contenu du message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Horodatage du message")

    @field_validator("id", mode="before")
    def validate_uuid(cls, v):
        try:
            uuid.UUID(str(v))
            return str(v)
        except Exception as e:
            raise ValueError("L'id doit être un UUID valide") from e

    @field_validator("sender")
    def validate_sender(cls, v):
        if v.lower() not in {"user", "assistant"}:
            raise ValueError("Le sender doit être 'user' ou 'assistant'")
        return v.lower()

    model_config = {
        "populate_by_name": True,
        "strip_whitespace": True,
    }
