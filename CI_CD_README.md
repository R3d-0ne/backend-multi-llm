# CI/CD Pipeline

Ce projet utilise GitHub Actions pour automatiser les tests, la construction et le déploiement de l'application FastAPI.

## Workflow CI/CD

Le pipeline CI/CD se déclenche à chaque push sur la branche `master` et inclut les étapes suivantes :

### 1. Versioning (Gestion des versions)
- Génère un identifiant de version basé sur le commit hash
- Crée le nom de l'image Docker avec le tag approprié
- Met à disposition ces informations pour les étapes suivantes

### 2. Unit Tests (Tests unitaires)
- Configuration Python 3.8
- Cache des dépendances pip pour optimiser les performances
- Installation des dépendances depuis `requirements.txt`
- Exécution des tests avec pytest

### 3. Build and Push (Construction et publication)
- Construction de l'image Docker
- Authentification sur GitHub Container Registry (GHCR)
- Publication de l'image avec le tag de version

### 4. API Tests (Tests d'API)
- Démarrage du conteneur avec l'image construite
- Tests de santé de l'API (endpoint `/docs`)
- Vérification que FastAPI fonctionne correctement

## Structure des tests

Le projet inclut plusieurs types de tests :

- `tests/test_utils.py` : Tests utilitaires et de base
- `tests/test_docker.py` : Tests spécifiques à Docker
- `tests/test_api_basic.py` : Tests de l'API FastAPI
- `tests/test_pipeline_autogen.py` : Tests du pipeline de traitement

## Variables d'environnement

Le CI/CD utilise les variables suivantes :

- `COMMIT_ID` : Injecté automatiquement avec l'ID du commit
- `GITHUB_TOKEN` : Token automatique pour l'authentification GHCR

## Endpoints ajoutés pour le CI/CD

- `/version` : Retourne la version de l'application et l'état de santé

## Exécution locale des tests

```bash
# Installer pytest
pip install pytest

# Exécuter tous les tests
PYTHONPATH=. pytest tests/ --maxfail=1 --disable-warnings --tb=short

# Exécuter un fichier de test spécifique
PYTHONPATH=. pytest tests/test_utils.py -v
```

## Construction locale Docker

```bash
# Construire l'image
docker build -t backend-multi-llm:latest .

# Exécuter le conteneur
docker run -p 8000:8000 backend-multi-llm:latest
```

L'application sera disponible sur http://localhost:8000