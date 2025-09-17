# Backend FastAPI – Recherche hybride, gestion de documents et génération LLM

Service FastAPI intégrant Qdrant pour la recherche vectorielle/hybride, un pipeline d’upload/traitement documentaire, la gestion de discussions/messages/contexte, et la génération de réponses via un LLM (Ollama/OpenAI-like).

## Démarrage rapide

- Dépendances Python: `pip install -r requirements.txt`
- Variables d’env: créez `backend/.env` (voir section Variables d’environnement)
- Lancer en local: `uvicorn app.main:app --reload`
- Avec Docker Compose: service `api` expose `http://localhost:8000`

## Endpoints principaux

- Santé et version:
  - `GET /` initialise/valide les collections Qdrant
  - `GET /health`
  - `GET /version`

- Recherche (`app/routes/search.py`):
  - `POST /search/` recherche avancée avec filtres, boost mots‑clés, rerank LLM, réponse optionnelle
  - `GET /search/simple` recherche simple par q=name avec fallback vectoriel
  - `POST /search/internal/` recherche + génération de réponse contextualisée
  - `POST /search/document-query/` QA ciblée sur un document par nom
  - Debug/tests: `GET /search/test-document-search/{document_name}`, `GET /search/test-simple-search/{query}`, `GET /search/debug-collection`

- Documents (`app/routes/documents.py`):
  - `POST /documents/` upload fichier, lance pipeline d’extraction (OCR, NER, résumé…) avec `use_llm_enrichment` (bool)
  - `GET /documents/{document_id}`
  - `GET /documents/?limit&offset`
  - `PATCH /documents/{document_id}` mise à jour métadonnées
  - `DELETE /documents/{document_id}`

- Discussions/messages/contexte/historique:
  - Discussions (`/discussions`, GET/POST/PUT/DELETE, `app/routes/discussions.py`)
  - Messages (`/messages`, POST/GET/PUT/DELETE, `app/routes/messages.py`)
  - Contextes (`/contexts`, POST/GET/DELETE, `app/routes/contexts.py`)
  - Historique (`/history/`, POST/GET/POST search/DELETE, `app/routes/history.py`)

- Génération LLM (`app/routes/generate.py`):
  - `POST /generate` {discussion_id?, settings_id?, current_message, additional_info?, model_id?}
  - `GET /models` liste des modèles disponibles (Ollama tags par défaut)
  - `POST /models/select?model_id=...` sélection du modèle par défaut

## Recherche hybride et reranking

- Collection cible: `documents` (format hybride supporté).
- Étapes: pré‑traitement requête (extraction d’entités/synonymes) → recherche Qdrant (détection auto standard/hybride, vecteur nommé `dense`) → filtres → boost mots‑clés → reranking LLM (scores 0‑10, fallback intelligent si parsing KO) → métriques et réponse optionnelle.
- Méthodes côté service: `SearchService.hybrid_search`, `simple_vector_search`, `search_document_by_name_*`, `search_with_generate_service`.

## Variables d’environnement (exemples)

- Qdrant:
  - `QDRANT_HOST` (ex: `qdrant` en Docker, `localhost` en local)
  - `QDRANT_PORT` (ex: `6333`)
- LLM (Ollama/OpenAI-like):
  - `LLM_API_URL` (défaut: `http://host.docker.internal:8000`)
  - `LLM_MODEL_NAME` (défaut: `llama3.1:8b`)
  - `LLM_MAX_TOKENS` (défaut: `1024`)
  - `LLM_TEMPERATURE` (défaut: `0.1`)
  - `LLM_REQUEST_TIMEOUT` (défaut: `10`)
- Divers:
  - `COMMIT_ID` pour `/version`

Placez ces valeurs dans `backend/.env` (chargé par Docker Compose).

## Lancement avec Docker

Compose (`docker-compose.yml` à la racine):
- `api`: build `./backend`, commande `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`, volume `./backend:/app`, ports `8000:8000`, dépend de `qdrant`.
- `qdrant`: image `qdrant/qdrant:latest`, ports `6333:6333`, volume nommé `qdrant_data`.
- `web`: Node 20, monte `./frontend`, lance `npm run dev -- --host 0.0.0.0 --port 5173`, ports `5173:5173`.

Backend Dockerfile installe Poppler et Tesseract pour OCR, modèles spaCy FR, et dépendances (voir `requirements.txt`).

## Exemples d’appels

- POST `/search/` minimal:
```json
{
  "query": "rib banque",
  "limit": 5,
  "use_llm_reranking": true,
  "boost_keywords": true,
  "generate_answer": false
}
```

- GET `/search/simple?q=document%20bancaire&limit=5`

- POST `/generate`:
```json
{
  "current_message": "Explique le contenu du document trouvé",
  "discussion_id": "<optionnel>",
  "settings_id": "<optionnel>",
  "model_id": "llama3.1:8b"
}
```

## Notes d’implémentation

- Les collections Qdrant sont (ré)initialisées au `GET /` via `create_or_update_collections` pour: `contexts`, `discussions`, `history`, `settings`, `messages`, `documents`.
- `DocumentUploadService` déclenche un pipeline (`libs/traitement_document`) après upload, avec OCR et enrichissement LLM optionnel.
- Le service LLM choisit automatiquement l’API chat/generate et gère des fallbacks robustes.