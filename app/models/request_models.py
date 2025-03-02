from pydantic import BaseModel, Field
from typing import Optional


class ContextRequest(BaseModel):
    """ Modèle pour ajouter un contexte """
    name: str = Field(..., title="Nom du contexte")
    content: str = Field(..., title="Contenu du contexte")


class QuestionRequest(BaseModel):
    discussion_id: Optional[str] = Field(None,
                                         title="ID de la discussion (peut être vide pour une nouvelle discussion)")
    context_id: str = Field(..., title="ID du contexte")
    question: str = Field(..., title="Texte de la question")


class DiscussionRequest(BaseModel):
    """ Modèle pour mettre à jour une discussion """
    context_id: str
    summary: str



class CreateDiscussionRequest(BaseModel):
    """Modèle pour la création d'une discussion"""
    context_id: str
    title: str

class SearchDiscussionRequest(BaseModel):
    """Modèle pour la recherche de discussions similaires"""
    query: str

class AddHistoryRequest(BaseModel):
    """Modèle pour ajouter un échange dans l'historique"""
    discussion_id: str
    question: str
    response: str

class SearchHistoryRequest(BaseModel):
    """Modèle pour la recherche d'échanges similaires"""
    query: str