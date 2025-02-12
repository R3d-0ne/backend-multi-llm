from pydantic import BaseModel, Field
from typing import Optional


class GenerateResponse(BaseModel):
    """ Réponse de l'IA """
    response: str = Field(..., title="Réponse générée par DeepSeek")


class ContextResponse(BaseModel):
    """ Contexte retourné après ajout """
    id: str = Field(..., title="ID du contexte")
    name: str = Field(..., title="Nom du contexte")
    content: str = Field(..., title="Contenu du contexte")


class HistoryEntry(BaseModel):
    """ Entrée d'historique """
    id: str = Field(..., title="ID de l'entrée d'historique")
    context_id: Optional[str] = Field(None, title="ID du contexte associé")
    question: str = Field(..., title="Question posée")
    response: str = Field(..., title="Réponse de l'IA")
    timestamp: str = Field(..., title="Date de l'interaction")
