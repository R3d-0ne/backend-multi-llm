import uuid
from datetime import datetime
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identifiant unique du document"
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Titre du document"
    )
    content: str = Field(
        ...,
        description="Contenu textuel complet du document"
    )
    entities: dict = Field(
        default_factory=dict,
        description="Entités extraites du document (ex: personnes, lieux)"
    )
    source: str = Field(
        ...,
        description="Chemin ou URL de la source du document"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date de création du document"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date de la dernière mise à jour du document"
    )

    model_config = {
        "populate_by_name": True,
        "strip_whitespace": True,
    }
